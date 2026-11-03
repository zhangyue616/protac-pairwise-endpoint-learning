"""R5 preparation control primitives. Standard library only; no model imports.

The draft is intentionally inactive. The future frozen contract, numeric budget,
resource admission and a new ROOT_GO must agree before a model worker is created.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
from r5_gpu_helper import GPUQueryClient, GPUQueryError

BASE = Path(__file__).resolve().parent
TICKS_PER_SECOND = 10_000_000


class ControlPause(RuntimeError):
    """Safe boundary stop, never an estimator FIT_FAILURE."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class ControlError(RuntimeError):
    """Identity, incomplete accounting or real failure requires diagnosis."""


def require(ok, reason):
    if not ok:
        raise ControlError(reason)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest().upper()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest().upper()


def target(path, root=BASE):
    path = Path(path).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'WRITE_OUTSIDE_R5_ROOT')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value, root=BASE):
    path = target(path, root)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
            allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_event(path, event, root=BASE):
    path = target(path, root)
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def validate_activation(contract, freeze, go, *, contract_sha, freeze_sha, go_sha):
    """Pure gate; callers must invoke this before scientific imports or Popen."""
    limit = contract.get('selected_total_model_process_budget_seconds')
    require(type(limit) is int and limit in (86400, 288000), 'NUMERIC_BUDGET_NOT_SELECTED')
    require(contract.get('status') == 'FROZEN_EXECUTION_CONTRACT', 'CONTRACT_NOT_FROZEN')
    require(contract.get('model_execution_authorized') is True, 'NO_EXECUTION_AUTHORITY')
    require(freeze.get('status') == 'FULL_EXECUTION_FREEZE', 'FULL_FREEZE_MISSING')
    require(freeze.get('contract_sha256') == contract_sha, 'FREEZE_CONTRACT_MISMATCH')
    require(freeze.get('queue_sha256') == contract.get('queue_sha256'), 'FREEZE_QUEUE_MISMATCH')
    require(freeze.get('downstream_adaptation_accepted') is True, 'DOWNSTREAM_NOT_FROZEN')
    require(bool(freeze.get('bindings')) and bool(freeze.get('decision_path')) and
        bool(freeze.get('decision_sha256')), 'FULL_FREEZE_BINDINGS_MISSING')
    require(go.get('status') == 'ROOT_GO' and go.get('contract_sha256') == contract_sha and
        go.get('freeze_sha256') == freeze_sha, 'NEW_ROOT_GO_REQUIRED')
    require(bool(go_sha) and go.get('budget_seconds') == limit and
        go.get('numeric_budget_decision_sha256') == freeze.get('decision_sha256'), 'GO_BUDGET_MISMATCH')
    require(go.get('serial_resource_slot_granted') is True and
        go.get('current_resource_preflight_sha256') and go.get('current_quota_evidence_sha256'),
        'CURRENT_RESOURCE_QUOTA_ADMISSION_REQUIRED')
    expected_limits = {'threads': 4, 'parent_soft_seconds': 1200,
        'minimum_available_ram_bytes': 3 * 1024**3, 'maximum_process_peak_working_set_bytes': 2 * 1024**3,
        'minimum_free_disk_bytes': 1024**3, 'minimum_cuda0_free_bytes': 4 * 1024**3}
    require(contract.get('resource_limits') == expected_limits, 'R5_RESOURCE_OR_SOFT_LIMIT_DRIFT')
    return limit


def activation_from_files(contract_path, expected_sha):
    contract_path = Path(contract_path).resolve()
    require(contract_path.parent == BASE and contract_path.name == 'EXECUTION_CONTRACT.json',
        'FUTURE_FROZEN_CONTRACT_PATH_REQUIRED')
    actual = digest(contract_path)
    require(actual == expected_sha.upper(), 'CONTRACT_SHA_MISMATCH')
    contract = read(contract_path)
    freeze_path, go_path = BASE / 'EXECUTION_FREEZE.json', BASE / 'ROOT_GO.json'
    require(freeze_path.is_file() and go_path.is_file(), 'FREEZE_AND_NEW_GO_REQUIRED')
    freeze, go = read(freeze_path), read(go_path)
    validate_activation(contract, freeze, go, contract_sha=actual,
        freeze_sha=digest(freeze_path), go_sha=digest(go_path))
    return contract, freeze, go


def accumulate_process_receipts(receipts):
    """Every attempt, including startup failure, is counted once using FILETIME."""
    total, scopes, seen = 0, {}, set()
    for item in receipts:
        identity = item['invocation_id']
        require(identity not in seen, 'DUPLICATE_PROCESS_RECEIPT')
        seen.add(identity)
        creation, exit_tick = item['creation_ticks'], item['exit_ticks']
        require(type(creation) is int and type(exit_tick) is int and creation > 0 and
            exit_tick >= creation, 'INVALID_KERNEL_PROCESS_CLOCK')
        require(type(item['elapsed_ticks']) is int and item['elapsed_ticks'] == exit_tick - creation,
            'KERNEL_CLOCK_ARITHMETIC_DRIFT')
        total += exit_tick - creation
        sid = item['scope_id']
        scopes[sid] = scopes.get(sid, 0) + exit_tick - creation
    return total, scopes


def prior_clock(root=BASE, exclude=None):
    folder = Path(root) / 'invocations'
    items = []
    for directory in sorted(folder.iterdir()) if folder.exists() else []:
        require(directory.is_dir(), 'UNKNOWN_INVOCATION_ENTRY')
        if exclude is not None and directory.resolve() == Path(exclude).resolve():
            continue
        receipt = directory / 'PROCESS_RECEIPT.json'
        require(receipt.is_file(), 'UNACCOUNTED_PROCESS_REQUIRES_EXACT_EXIT_RECONCILIATION')
        item = read(receipt)
        require(item['invocation_id'] == directory.name, 'INVOCATION_ID_DRIFT')
        start_path = directory / 'PROCESS_START.json'
        require(start_path.is_file(), 'PROCESS_START_MISSING')
        start = read(start_path)
        require(start['pid'] == item['pid'] and start['scope_id'] == item['scope_id'] and
            start['creation_ticks'] == item['creation_ticks'] and start['contract_sha256'] == item['contract_sha256'],
            'PROCESS_RECEIPT_START_IDENTITY_DRIFT')
        items.append(item)
    return accumulate_process_receipts(items)


def boundary_decision(*, total_ticks, scope_ticks, limit_seconds, limits, resource,
        cuda=False, user_stop=False, controller_stop=False, quota=None):
    """Pure boundary calculation. Unknown quota is recorded; never guessed."""
    if user_stop:
        return 'USER_PAUSE'
    if controller_stop:
        return 'CONTROLLER_PAUSE'
    if quota is not None:
        remaining = quota.get('ordinary_remaining_percent')
        if quota.get('reliable') is True and type(remaining) in (int, float) and remaining < 20:
            return 'QUOTA_BELOW20_PAUSE'
    if total_ticks >= limit_seconds * TICKS_PER_SECOND:
        return 'TOTAL_BUDGET_EXHAUSTED'
    if scope_ticks >= limits['parent_soft_seconds'] * TICKS_PER_SECOND:
        return 'PARENT_SOFT1200_EXHAUSTED'
    if resource is None:
        return None  # Cheap stop/clock check before starting an expensive resource query.
    if resource['available_ram_bytes'] < limits['minimum_available_ram_bytes']:
        return 'AVAILABLE_RAM_BELOW3GIB'
    if resource['free_disk_bytes'] < limits['minimum_free_disk_bytes']:
        return 'DISK_BELOW1GIB'
    if resource['process_peak_working_set_bytes'] > limits['maximum_process_peak_working_set_bytes']:
        return 'PROCESS_PEAK_RSS_ABOVE2GIB'
    if cuda and resource['cuda0_free_bytes'] < limits['minimum_cuda0_free_bytes']:
        return 'CUDA0_FREE_BELOW4GIB'
    return None


def fit_key(context):
    fields = ('scope_id', 'phase', 'candidate_id', 'inner_fold', 'backfit_iteration')
    return tuple(context[field] for field in fields)


def validate_fit_ledger(rows, contract_sha):
    attempts, completes, failures = {}, {}, []
    for row in rows:
        require(row.get('contract_sha256') == contract_sha, 'MIXED_FIT_CONTRACT')
        if row['event'] not in ('FIT_ATTEMPT', 'FIT_COMPLETE', 'FIT_FAILURE'):
            raise ControlError('UNKNOWN_FIT_EVENT')
        key = fit_key(row)
        if row['event'] == 'FIT_ATTEMPT':
            require(key not in attempts, 'DUPLICATE_FIT_ATTEMPT')
            attempts[key] = row
        elif row['event'] == 'FIT_COMPLETE':
            require(key in attempts and key not in completes, 'FIT_COMPLETE_WITHOUT_UNIQUE_ATTEMPT')
            require(row.get('dependency') == attempts[key].get('dependency'), 'FIT_DEPENDENCY_LEDGER_DRIFT')
            completes[key] = row
        else:
            failures.append(row)
    require(not failures, 'FIT_FAILURE_NO_BLIND_RETRY')
    require(set(attempts) == set(completes), 'UNCLOSED_FIT_NO_BLIND_RETRY')
    return completes


def verify_completed_fit(receipt, event, dependency, artifacts):
    """Pure mockable comparison before any checkpoint model is loaded."""
    require(receipt.get('dependency') == dependency == event.get('dependency'), 'EXACT_FIT_IDENTITY_DRIFT')
    require(artifacts.get('receipt_sha256') == event.get('receipt_sha256'), 'FIT_RECEIPT_SHA_DRIFT')
    require(artifacts.get('model_sha256') == receipt.get('model_sha256'), 'MODEL_SHA_DRIFT')
    require(artifacts.get('training_prediction_sha256') == receipt.get('training_prediction_sha256'),
        'TRAINING_VECTOR_SHA_DRIFT')


def checkpoint_evidence(scope):
    folder = BASE / 'parent_slots' / scope['scope_id']
    paths = [folder / name for name in ('PARENT_IDENTITY.json', 'FIT_LEDGER.jsonl', 'OUTPUT_BINDINGS.json', 'RESULTS.json')]
    if scope['kind'] not in ('internal_parent', 'source_parent'):
        paths += [BASE / name for name in ('PARENT_COMPLETION_MANIFEST.json', 'SOURCE_COMPLETION_SELECTION.json',
            'ripk1_case/CHECKPOINT_BINDINGS.json', 'ripk1_case/FROZEN_MANIFEST.json', 'ripk1_case/PREDICTION_STARTED.json',
            'ripk1_case/MODEL_OPERATIONS.jsonl', 'ripk1_case/RESULTS_RECEIPT.json',
            'ripk1_case/FREEZE_ACTUAL_OPERATION_ACCOUNTING.json','ripk1_case/PREDICTION_ACTUAL_OPERATION_ACCOUNTING.json')]
    return {path.as_posix(): digest(path) for path in paths if path.is_file()}


def kernel_times(handle=None):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessTimes.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
    kernel.GetProcessTimes.restype = ctypes.c_int
    values = [ctypes.c_ulonglong() for _ in range(4)]
    actual = kernel.GetCurrentProcess() if handle is None else handle
    require(kernel.GetProcessTimes(actual, *(ctypes.byref(v) for v in values)), 'KERNEL_TIMES_UNAVAILABLE')
    return dict(zip(('creation_ticks', 'exit_ticks', 'kernel_ticks', 'user_ticks'), [v.value for v in values]))


def now_ticks():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    value = ctypes.c_ulonglong()
    kernel.GetSystemTimePreciseAsFileTime.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    kernel.GetSystemTimePreciseAsFileTime(ctypes.byref(value))
    return value.value


def _host_resources(root):
    class Memory(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong)
            for n in ('totalPhys', 'availPhys', 'totalPage', 'availPage', 'totalVirtual', 'availVirtual', 'availExtended')]
    class ProcessMemory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('faults', ctypes.c_ulong)] + [(n, ctypes.c_size_t)
            for n in ('peakWorkingSet', 'workingSet', 'quotaPeakPaged', 'quotaPaged', 'quotaPeakNonPaged',
                'quotaNonPaged', 'pagefile', 'peakPagefile')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    memory, process = Memory(), ProcessMemory()
    memory.length, process.cb = ctypes.sizeof(memory), ctypes.sizeof(process)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong]
    require(kernel.GlobalMemoryStatusEx(ctypes.byref(memory)), 'RAM_QUERY_FAILED')
    require(psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(process), process.cb), 'RSS_QUERY_FAILED')
    value = {'available_ram_bytes': memory.availPhys, 'free_disk_bytes': shutil.disk_usage(root).free,
        'process_peak_working_set_bytes': process.peakWorkingSet}
    return value


def resources(root=BASE, cuda=False, *, gpu_query=None):
    value = _host_resources(root)
    if cuda:
        if gpu_query is not None:
            value['cuda0_free_bytes'] = gpu_query()
        else:
            # Unchanged driver preflight signature: one-shot ownership, never a global helper.
            client = GPUQueryClient('one-shot-resource-preflight')
            try:
                value['cuda0_free_bytes'] = client.query()
            except BaseException as query_error:
                try:
                    client.close()
                except BaseException as cleanup_error:
                    combined = GPUQueryError('RESOURCE_QUERY_PRIMARY_ERROR: ' + repr(query_error) +
                        '; SECONDARY_HELPER_CLEANUP_ERROR: ' + repr(cleanup_error))
                    combined.original_query_error = query_error
                    combined.helper_cleanup_error = cleanup_error
                    raise combined from query_error
                raise
            else:
                client.close()
    return value


class FrozenBytes:
    """Once per driver full verification, metadata check per subsequent worker."""
    def __init__(self, bindings):
        self.bindings = bindings

    @staticmethod
    def metadata(path):
        stat = Path(path).stat()
        return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'ctime_ns': stat.st_ctime_ns}

    def verify(self, full=False):
        for name, item in self.bindings.items():
            current = self.metadata(name)
            if full or current != item['metadata']:
                require(digest(name) == item['sha256'], 'FROZEN_INPUT_CODE_SHA_DRIFT: ' + name)


class Runtime:
    def __init__(self, contract, scope, invocation, contract_sha):
        self.contract, self.scope, self.invocation = contract, scope, Path(invocation)
        self.contract_sha = contract_sha
        self.prior_total, prior_scopes = prior_clock(exclude=self.invocation)
        self.prior_scope = prior_scopes.get(scope['scope_id'], 0)
        self.creation = kernel_times()['creation_ticks']
        self.atomic_fit = False
        interval = contract.get('runtime_clock_observation_interval_seconds', 5.0)
        require(type(interval) in (int, float) and math.isfinite(interval) and 0.1 <= interval <= 60,
            'RUNTIME_CLOCK_OBSERVATION_INTERVAL_INVALID')
        self.observation_interval_ticks = int(interval * TICKS_PER_SECOND)
        self.last_observation_tick = None
        self.gpu_client = None

    def observation(self):
        tick = now_ticks()
        return {'scope_id': self.scope['scope_id'], 'creation_ticks': self.creation, 'observed_ticks': tick,
            'total_elapsed_ticks': self.prior_total + tick - self.creation,
            'scope_elapsed_ticks': self.prior_scope + tick - self.creation}

    def _reason(self, value, resource=None):
        marker = Path(self.contract['shared_user_stop_path'])
        quota_path = BASE / 'CURRENT_QUOTA_EVIDENCE.json'
        quota = read(quota_path) if quota_path.exists() else None
        return boundary_decision(total_ticks=value['total_elapsed_ticks'], scope_ticks=value['scope_elapsed_ticks'],
            limit_seconds=self.contract['selected_total_model_process_budget_seconds'], limits=self.contract['resource_limits'],
            resource=resource, cuda=self.scope.get('learner') == 'xgboost', user_stop=marker.exists(),
            controller_stop=(BASE / 'CONTROLLER_STOP_REQUEST.json').exists(), quota=quota)

    def _gpu_query(self):
        if self.gpu_client is None:
            self.gpu_client = GPUQueryClient('worker:' + self.invocation.name)
        try:
            return self.gpu_client.query()
        except GPUQueryError as exc:
            raise ControlError(str(exc)) from exc

    def write_observation(self, value, resource=None, reason=None, *, force=False, error=None, event='boundary'):
        # This observer is informational; prior_clock/PROCESS_RECEIPT remain authoritative.
        if not force and self.last_observation_tick is not None and (
                value['observed_ticks'] - self.last_observation_tick < self.observation_interval_ticks):
            return
        atomic_json(self.invocation / 'RUNTIME_CLOCK.json', {**value, 'resource': resource,
            'stop_reason': reason, 'control_error': error, 'observation_event': event,
            'informational_only': True})
        self.last_observation_tick = value['observed_ticks']

    def _raise_stop(self, reason, value, resource=None, *, error=None):
        pause = ControlPause(reason)
        if error is not None:
            pause.control_error = repr(error)
        try:
            self.write_observation(value, resource, reason, force=True, error=repr(error) if error else None)
        except BaseException as exc:
            pause.observation_error = repr(exc)
        raise pause

    def boundary(self):
        require(not self.atomic_fit, 'BOUNDARY_INSIDE_ATOMIC_FIT')
        value = self.observation()
        reason = self._reason(value)
        if reason:
            self._raise_stop(reason, value)  # User stop is checked without creating/querying a helper.
        resource = None
        try:
            resource = resources(cuda=self.scope.get('learner') == 'xgboost', gpu_query=self._gpu_query)
        except BaseException as exc:
            value = self.observation()  # Include failed query and its synchronous cleanup time.
            reason = self._reason(value)
            if reason:
                self._raise_stop(reason, value, error=exc)
            try:
                self.write_observation(value, force=True, error=repr(exc), event='error')
            except BaseException as writer:
                exc.observation_error = repr(writer)
            raise
        value = self.observation()  # Admission must use post-query wall clock, never its old sample.
        reason = self._reason(value, resource)
        if reason:
            self._raise_stop(reason, value, resource)
        try:
            self.write_observation(value, resource)
        except BaseException as exc:
            value = self.observation()
            reason = self._reason(value, resource)
            if reason:
                self._raise_stop(reason, value, resource, error=exc)
            raise ControlError('RUNTIME_CLOCK_OBSERVATION_WRITE_FAILED: ' + repr(exc)) from exc
        value = self.observation()  # fsync/replace and new stop requests are charged before any next fit.
        reason = self._reason(value, resource)
        if reason:
            self._raise_stop(reason, value, resource)
        return value

    def close(self):
        if self.gpu_client is not None:
            self.gpu_client.close()
