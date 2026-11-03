"""Reconstruct the final curated main-analysis inputs from the fixed TACK snapshot.

Download the source separately. The repository supplies only author-generated
row/ID membership, source references, fixed assignments and previously adopted
local graph edits.
No source SMILES, complete corrected structures or measurements are embedded.
"""
from __future__ import annotations

import io
import argparse
from argparse import Namespace
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CURATION = HERE / 'curation'
SNAPSHOT_SHA256 = '2893CE64FE1D5AD50F6BF97AA63FDE9597D11E6A1B388CFE30A7CFC79BB06875'
SNAPSHOT_URL = 'https://huggingface.co/datasets/ailab-bio/TACK/resolve/0ebfb3627cfeee826c3392586d46940402237b11/DC50/train-00000-of-00001.parquet'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest().upper()


def canonical(molecule):
    from rdkit import Chem
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def apply_graph_edit(before, rule):
    """Replay the adopted atom additions and absolute bridgehead assignments."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler
    if text_digest(canonical(before)) != rule['before_identity_hash']:
        raise RuntimeError('source graph differs before the fixed edit')
    heads = rule['bridgeheads_source_atom_indices_0_based']
    edit = Chem.RWMol(before)
    bridge = edit.AddAtom(Chem.Atom(6))
    for index in heads:
        if before.GetAtomWithIdx(index).GetSymbol() != 'C' or before.GetAtomWithIdx(index).GetTotalNumHs() != 2:
            raise RuntimeError('fixed bridgehead is not the recorded source CH2')
        edit.AddBond(index, bridge, Chem.BondType.SINGLE)
    methyl = rule['N_methyl_site']
    if methyl is not None:
        # The published rule stores the original source atom index only.
        nitrogen = int(methyl)
        atom = edit.GetAtomWithIdx(nitrogen)
        if atom.GetSymbol() != 'N' or not atom.GetIsAromatic() or atom.GetNumExplicitHs() != 1:
            raise RuntimeError('fixed N-methyl site differs from the recorded aromatic [nH]')
        atom.SetNumExplicitHs(0)
        added = edit.AddAtom(Chem.Atom(6))
        edit.AddBond(nitrogen, added, Chem.BondType.SINGLE)
    molecule = edit.GetMol()
    Chem.SanitizeMol(molecule)
    wanted = rule['bridgehead_absolute_CIP']
    selected = []
    for tags in itertools.product((Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                   Chem.ChiralType.CHI_TETRAHEDRAL_CCW), repeat=2):
        trial = Chem.Mol(molecule)
        for index, tag in zip(heads, tags):
            trial.GetAtomWithIdx(index).SetChiralTag(tag)
        Chem.AssignStereochemistry(trial, cleanIt=True, force=True)
        if rule['CIP_assignment'] == 'rdCIPLabeler':
            rdCIPLabeler.AssignCIPLabels(trial)
        labels = [trial.GetAtomWithIdx(index).GetProp('_CIPCode') for index in heads]
        if labels == wanted:
            selected.append(trial)
    if len(selected) != 1 or text_digest(canonical(selected[0])) != rule['after_identity_hash']:
        raise RuntimeError('fixed graph edit did not reproduce its recorded final identity')
    return selected[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True, type=Path,
                        help='separately obtained fixed TACK parquet; its SHA is checked before reading')
    parser.add_argument('--output', required=True, type=Path,
                        help='new private output directory outside the repository')
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(HERE.parents[1]):
        parser.error('--output must be a new directory outside the repository')
    if digest(args.snapshot) != SNAPSHOT_SHA256:
        raise RuntimeError('TACK snapshot bytes differ from the fixed input identity')
    manifest = json.loads((CURATION / 'index_manifest.json').read_text('utf-8'))
    for name, expected in manifest['files'].items():
        if digest(CURATION / name) != expected:
            raise RuntimeError('author curation index identity mismatch: ' + name)
    import numpy as np
    import pandas as pd
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    if rdBase.rdkitVersion != '2026.03.3':
        raise RuntimeError('the frozen structure reconstruction requires RDKit 2026.03.3')
    source = pd.read_parquet(args.snapshot)
    if len(source) != 4184:
        raise RuntimeError('fixed snapshot row count differs')
    index = pd.read_csv(CURATION / 'record_index.csv', keep_default_na=False)
    pair_index = pd.read_csv(CURATION / 'pair_index.csv', keep_default_na=False)
    if len(index) != 699 or len(pair_index) != 5804 or not index.record_id.is_unique or not pair_index.pair_id.is_unique:
        raise RuntimeError('curation index cardinality differs')
    graph_document = json.loads((CURATION / 'graph_edits.json').read_text('utf-8'))
    rules = {int(x['example']): x for x in graph_document['rules']}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False,
        useBondTypes=True, onlyNonzeroInvariants=False, includeRingMembership=True, includeRedundantEnvironments=False)
    records, fingerprints, bitvalues = [], [], {}
    for member in index.itertuples(index=False):
        original = source.iloc[int(member.raw_index)]
        value = float(original['Value'])
        if not math.isfinite(value) or value <= 0 or str(original['Value_Unit']).lower() != 'nm':
            raise RuntimeError('fixed curated endpoint is not a positive nM point')
        molecule = Chem.MolFromSmiles(str(original['SMILES']))
        if molecule is None:
            raise RuntimeError('fixed curated source structure failed parsing')
        if str(member.graph_edit_example):
            rule = rules[int(member.graph_edit_example)]
            if int(member.raw_index) not in rule['raw_indices']:
                raise RuntimeError('graph edit source-row membership mismatch')
            molecule = apply_graph_edit(molecule, rule)
        smiles = canonical(molecule)
        if text_digest(smiles) != member.identity_hash:
            raise RuntimeError('final reconstructed identity mismatch: ' + member.record_id)
        records.append({'record_id': member.record_id, 'identity_hash': member.identity_hash,
            'canonical_isomeric_smiles_full': smiles, 'normalized_poi_id': member.normalized_poi_id,
            'pair_context_hash': member.pair_context_hash, 'pdc50_median': 9.0 - math.log10(value),
            'primary_pair_eligible': True})
        bits = generator.GetFingerprint(molecule).GetOnBits()
        integer = sum(1 << bit for bit in bits)
        bitvalues[member.record_id] = integer
        fingerprints.append({'record_id': member.record_id, 'fp_hex': format(integer, '0512X'),
                             'popcount': len(bits)})
    records = pd.DataFrame(records).sort_values('record_id', kind='mergesort').reset_index(drop=True)
    # The study's input build read pDC50 back through the pandas 3.0 default CSV parser before forming pairs.
    records['pdc50_median'] = pd.read_csv(io.StringIO(records[['pdc50_median']].to_csv(index=False))).pdc50_median
    by_id = records.set_index('record_id')
    listed = {frozenset((x.record_id_i,x.record_id_j)): x for x in pair_index.itertuples(index=False)}
    if len(listed) != 5804:
        raise RuntimeError('duplicate endpoint pair in curation index')
    pairs, generated = [], set()
    for _, group in records.groupby(['normalized_poi_id','pair_context_hash'], sort=True):
        for a,b in itertools.combinations(group.itertuples(index=False),2):
            if a.identity_hash == b.identity_hash:
                continue
            left,right = bitvalues[a.record_id],bitvalues[b.record_id]
            similarity = (left & right).bit_count() / (left | right).bit_count()
            if similarity < 0.5 - 1e-12:
                continue
            key = frozenset((a.record_id,b.record_id))
            if key not in listed:
                raise RuntimeError('reconstructed compatible edge is outside the frozen pair index')
            entry = listed[key]
            i,j = by_id.loc[entry.record_id_i],by_id.loc[entry.record_id_j]
            y = float(j.pdc50_median - i.pdc50_median)
            pairs.append({'pair_id': entry.pair_id, 'record_id_i': entry.record_id_i,
                'record_id_j': entry.record_id_j, 'identity_hash_i': i.identity_hash,
                'identity_hash_j': j.identity_hash, 'normalized_poi_id': i.normalized_poi_id,
                'pair_context_hash': i.pair_context_hash, 'true_delta': -y,
                'is_cliff': abs(y) >= 1.0-1e-12, 'canonical_target_b_minus_a': y, 'tanimoto': similarity})
            generated.add(key)
    if generated != set(listed):
        raise RuntimeError('the complete reconstructed edge population differs from the frozen pair index')
    # Intermediate tables remain private, next to the requested output.
    tables = output.with_name(output.name + '_curated_tables')
    if tables.exists():
        raise RuntimeError('private intermediate directory already exists')
    tables.mkdir(parents=True)
    records.to_csv(tables / 'records_revised.csv', index=False, lineterminator='\n')
    pd.DataFrame(pairs).sort_values('pair_id').to_csv(tables / 'pairs_revised.csv', index=False, lineterminator='\n')
    pd.DataFrame(fingerprints).sort_values('record_id').to_csv(tables / 'fingerprints.csv', index=False, lineterminator='\n')
    index[['record_id','c2_source_id']].to_csv(tables / 'record_sources.csv', index=False, lineterminator='\n')
    from prepare_inputs import prepare
    prepare(Namespace(records=tables/'records_revised.csv', pairs=tables/'pairs_revised.csv',
        fingerprints=tables/'fingerprints.csv', outer_folds=CURATION/'outer_identity_folds.csv',
        inner_folds=CURATION/'inner_identity_folds.csv', source_inner_folds=CURATION/'source_inner_identity_folds.csv',
        record_sources=tables/'record_sources.csv', selected_sources=CURATION/'selected_sources.json',
        output=output), verify_original_tables=False)
    receipt = {'status': 'FIXED_SOURCE_TO_CURATED_MAIN_INPUTS_RECONSTRUCTED',
        'snapshot_sha256': SNAPSHOT_SHA256, 'snapshot_url': SNAPSHOT_URL,
        'curation_index_sha256': digest(CURATION/'index_manifest.json'),
        'source_rows': len(source), 'records': len(records), 'identities': records.identity_hash.nunique(),
        'compatible_pairs': len(pairs), 'graph_edit_examples': sorted(rules),
        'fits': 0, 'model_predictions': 0, 'upstream_database_to_TACK_history_reconstructed': False,
        'scope': 'final main-analysis scientific input fields; source text and ancillary audit columns are intentionally omitted'}
    (output/'RECONSTRUCTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
