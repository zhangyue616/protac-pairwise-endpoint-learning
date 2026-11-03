"""Bounded table/pointer extraction from accepted artifacts; no fit or prediction."""
from pathlib import Path
import csv
import hashlib
import json
import re
from collections import defaultdict
from statistics import fmean

OUT = Path(__file__).resolve().parent
ROOT = Path('/LOCAL_PROJECT_ROOT')
INTAKE = ROOT/'intake'
CAND = INTAKE/'remaining_local_preparation_20261012/candidate_repair'
R6 = CAND/'r6_budget_b_20261012/summary'
R5 = ROOT/'r5_analysis'
MAIN = INTAKE/'revision_r2/source/Main_Article.md'
SI = INTAKE/'revision_r1/source/Supporting_Information.md'


def readj(path):
    return json.loads(path.read_text('utf-8-sig'))


def readc(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def writej(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def writec(name, rows):
    if not rows:
        raise ValueError(name+' cannot be empty')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/name).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def r6(name, scenario='corrected_shadow'):
    return [row for row in readc(R6/'results'/name) if row['scenario'] == scenario]


def tables(path):
    lines = path.read_text('utf-8').splitlines()
    result = []
    for i, line in enumerate(lines):
        match = re.match(r'^(?:### |\*\*)Table (S?\d+)\. (.*?)(?:\*\*)?$', line)
        if not match:
            continue
        j = i+1
        while j < len(lines) and not lines[j].startswith('|'):
            j += 1
        start = j
        while j < len(lines) and lines[j].startswith('|'):
            j += 1
        raw = lines[start:j]
        parsed = [[v.strip() for v in row.strip('|').split('|')] for row in raw]
        result.append({'table':match[1], 'title':match[2], 'source':str(path),
                       'heading_line':i+1, 'first_row_line':start+1, 'last_row_line':j,
                       'columns':parsed[0], 'rows':parsed[2:]})
    return result


def main():
    expected = {MAIN:'A001F4BE21666262F7261EA4A7C0FCBAB935113D70A07B95F59A70C0B94CF9E1',
                SI:'FFA20E77A8B14E66A06E4D0F8E59222338BBB370532FD622CEFC77C4FA7D9134'}
    for path, sha in expected.items():
        if hashlib.sha256(path.read_bytes()).hexdigest().upper() != sha:
            raise RuntimeError('Specified baseline source drift: '+str(path))

    accepted = readj(CAND/'CANDIDATE_ACCEPTANCE.json')
    patches = readj(CAND/'CANDIDATE_PATCHES.json')['scenarios'][0]['patches']
    anchors = {row['example']:row for row in readj(ROOT/'remaining_preparation/patent/SEVEN_CANDIDATE_SCHEME_ANCHORS.json')['candidates']}
    ga = readj(CAND/'group_a/CANDIDATES.json')
    gb = readj(CAND/'group_b/CANDIDATES.json')
    source_pdfs = readj(ROOT/'remaining_preparation/patent/SOURCE_IDENTITY.json')
    corrected_records = readc(CAND/'r5_dependency_preparation/candidate_r5_contract/input_build/data/records_revised.csv')
    source_rows, source_full = [], []
    for patch in patches:
        ex = patch['example']
        anchor = anchors[ex]
        current = [r for r in corrected_records if r['source_row_ids'] in patch['source_row_ids']]
        assert len(current)==len(patch['record_ids']) and {r['identity_hash'] for r in current}=={patch['candidate_identity_hash']}
        crops = []
        if ex in (17,20,36):
            crops = [row for row in ga['viewed_source_images'] if row['example']==ex]
        elif ex in (39,56,61):
            item = next(row for row in gb['examples'] if row['example']==ex)
            crops = item['original_final_product_anchors']
        full = {**patch, 'source_patent':'WO2024183650A1 / WO2024183650A9',
                'record_ids_scope':'historical before-repair IDs from accepted patch contract',
                'corrected_record_ids':[r['record_id'] for r in current],
                'corrected_record_source_bindings':[{'source_row_ids':r['source_row_ids'],'record_id':r['record_id'],'identity_hash':r['identity_hash']} for r in current],
                'source_pdf_identity':source_pdfs,
                'A1_pdf_page_1_based':anchor['a1_final_product_page'],
                'A9_pdf_page_1_based':anchor['a9_final_product_page'],
                'source_full_page_images':[str(ROOT/'remaining_preparation/patent'/anchor[k]) for k in ('a1_image','a9_image')],
                'source_crop_pointers':crops,
                'source_acceptance':str(CAND/'CANDIDATE_ACCEPTANCE.json'),
                'source_acceptance_status':accepted['status'],
                'status_precedence':'Final CANDIDATE_ACCEPTANCE supersedes provisional/pending flags in producer candidate files',
                'scope_limit':'Source-drawn complete constitution and depicted bridge stereochemistry; no measurement/assay certification',
                'RIPK1_history_is_separate':True}
        source_full.append(full)
        source_rows.append({'example':ex,'TPD_ID':patch['tpd_id'],'affected_records':len(patch['record_ids']),
                            'source_row_ids':';'.join(patch['source_row_ids']),
                            'corrected_record_ids':';'.join(r['record_id'] for r in current),
                            'change':patch['edit_summary'],'bridgehead_configuration':'R,R',
                            'stereo_evidence':'source hashed-geometry decode' if ex in (36,56,61) else 'explicit source R labels',
                            'A1_PDF_page':anchor['a1_final_product_page'],'A9_PDF_page':anchor['a9_final_product_page'],
                            'candidate_evidence':patch['candidate_evidence']['path'],
                            'selector':patch['candidate_evidence']['selector'],
                            'complete_mapping':'S30_repair_candidates.json',
                            'scope_limit':'Graph/stereobond fidelity; measurements and assay comparability not certified'})
    assert len(source_rows)==7 and sum(row['affected_records'] for row in source_rows)==12
    writec('S30_repair_candidates.csv', source_rows)
    writej('S30_repair_candidates.json',{'identities':7,'records':12,'source_acceptance':accepted['status'],
            'separate_RIPK1_history':{'source':str(SI),'line':1407,'event':'Seven historical ring-closure transcription errors corrected before case freeze; not these seven training identities'},
            'source_unspecified_stereo':'Preserved, including glutarimide; Example61 modern potential centers remain unspecified at source indices 5,27,30,34; do not state exactly one without legacy-detector qualification',
            'candidates':source_full})

    macro = r6('MACRO_METRICS.csv')
    macro_zero = fmean(float(r['zero_mae']) for r in r6('ZERO_BY_GROUP.csv'))
    pmap = {'direct_pair':'Direct','endpoint_E0':'E0','endpoint_E0_matched':'E0-matched','endpoint_E1':'E1'}
    lname = {'ridge':'Ridge','random_forest':'RF','xgboost':'XGB'}
    summary = r6('LEARNER_CONTRAST_SUMMARY.csv')
    out = []
    for rep in ('binary','count'):
        for learner in ('ridge','random_forest','xgboost'):
            part = {row['pipeline']:row for row in macro if row['representation']==rep and row['learner']==learner}
            row = {'Representation':rep,'Learner':lname[learner]}
            row.update({pmap[arm]+'_MAE':part[arm]['group_macro_mae'] for arm in pmap})
            row['Zero_MAE'] = macro_zero
            row['E1_minus_E0matched'] = next(r['theta_E1_minus_E0matched'] for r in summary if r['representation']==rep and r['learner']==learner)
            row['source']='MACRO_METRICS.csv + LEARNER_CONTRAST_SUMMARY.csv; scenario=corrected_shadow'
            out.append(row)
    writec('R6_Table3.csv',out)
    pooled = r6('POOLED_METRICS.csv')
    pooled_means = {}
    for rep in ('binary','count'):
        values = []
        for learner in ('ridge','random_forest','xgboost'):
            part = {r['pipeline']:r for r in pooled if r['representation']==rep and r['learner']==learner}
            values.append(float(part['endpoint_E1']['pair_pooled_mae'])-float(part['endpoint_E0_matched']['pair_pooled_mae']))
        pooled_means[rep] = fmean(values)
    writej('R6_reading_summary.json',{'source':'Accepted corrected_shadow CSV rows; fixed-table arithmetic only',
            'primary':r6('PRIMARY_SUMMARY.csv'),'macro_zero_mae':macro_zero,
            'macro_cells_better_than_zero':{pmap[arm]:sum(float(r['group_macro_mae'])<macro_zero for r in macro if r['pipeline']==arm) for arm in pmap},
            'pooled_three_learner_mean_E1_minus_E0matched':pooled_means,
            'pooled_metrics':pooled,'scale_diagnostic':r6('SCALE_DIAGNOSTIC_ALL_43.csv')})
    points = r6('GROUP_AVERAGE_CONTRASTS.csv')
    assert len(points)==86
    writec('R6_Figure4_points.csv',points)
    census = {row['component_id']:row for row in readc(R6/'corrected_metadata/group_census.csv')}
    support = {row['dependency_component']:row for row in readc(R6/'corrected_metadata/group_support.csv')}
    zeros = {row['heldout_component_id']:row for row in r6('ZERO_BY_GROUP.csv')}
    normalized_sources = defaultdict(set)
    for row in readc(CAND/'impact/run_source_graphs_20261012/repair_seven_source_graphs/record_source_group_map.csv'):
        normalized_sources[row['component_id']].add(row['normalized_source_id'])
    s26=[]
    for comp in sorted(support):
        row={'component':comp,**census[comp],**support[comp],**zeros[comp],
             'n_normalized_sources':len(normalized_sources[comp])}
        s26.append(row)
    writec('R6_S26_support.csv',s26)
    s27=[]
    for comp in sorted(support):
        row={'Component':comp}
        for rep in ('binary','count'):
            point=next(p for p in points if p['heldout_component_id']==comp and p['representation']==rep)
            row['d_'+rep]=point['delta_mae']
            row['u_'+rep]=point['bounded_relative_difference_u']
        s27.append(row)
    writec('R6_S27_contrasts.csv',s27)
    primary = r6('PRIMARY_SUMMARY.csv')
    scale = r6('SCALE_DIAGNOSTIC_ALL_43.csv')
    learner_contrasts = r6('GROUP_LEARNER_CONTRASTS.csv')
    s28=[]
    for rep in ('binary','count'):
        for learner in ('ridge','random_forest','xgboost'):
            values=[float(r['delta_mae']) for r in learner_contrasts if r['representation']==rep and r['learner']==learner]
            row=next(r for r in summary if r['representation']==rep and r['learner']==learner)
            s28.append({'representation':rep,'learner':lname[learner],
                         'theta':row['theta_E1_minus_E0matched'],'E1_better':sum(v < -1e-12 for v in values),
                         'ties':sum(abs(v)<=1e-12 for v in values),'E0matched_better':sum(v > 1e-12 for v in values),
                         'rho_S_d':'','rho_S_u':''})
        row=next(r for r in primary if r['representation']==rep)
        s28.append({'representation':rep,'learner':'Three-learner mean','theta':row['theta_E1_minus_E0matched'],
                    'E1_better':row['groups_E1_better'],'ties':row['groups_tied'],'E0matched_better':row['groups_E0matched_better'],
                    'rho_S_d':row['spearman_support_delta'],
                    'rho_S_u':next(r['spearman_support_bounded_relative_difference'] for r in scale if r['representation']==rep)})
    writec('R6_S28_summary.csv',s28)
    sensitivities = r6('SENSITIVITY.csv')
    s29=[]
    for rep in ('binary','count'):
        p=next(r for r in primary if r['representation']==rep)
        s29.append({'representation':rep,'population':'all_43_components','components':'43',
                    'theta_min':p['theta_E1_minus_E0matched'],'theta_max':p['theta_E1_minus_E0matched'],
                    'rho_min':p['spearman_support_delta'],'rho_max':p['spearman_support_delta'],'undefined_rhos':0})
        for scope in ('at_least_5_pair_endpoint_identities','delete_one_component_fixed_predictions','omit_two_largest_pair_groups'):
            part=[r for r in sensitivities if r['representation']==rep and r['scope']==scope]
            vals=[float(r['theta_E1_minus_E0matched']) for r in part]
            rhos=[float(r['spearman_support_delta']) for r in part if r['spearman_support_delta']!='']
            s29.append({'representation':rep,'population':scope,'components':part[0]['components'],
                        'theta_min':min(vals),'theta_max':max(vals),'rho_min':min(rhos) if len(rhos)==len(part) else '',
                        'rho_max':max(rhos) if len(rhos)==len(part) else '', 'undefined_rhos':len(part)-len(rhos)})
    writec('R6_S29_sensitivity.csv',s29)

    mt,st=tables(MAIN),tables(SI)
    writej('S31_old_tables.json',{'baseline_source_SHA':{str(k):v for k,v in expected.items()},
                                 'purpose':'Historical baseline only; no current evidence accepted by extraction',
                                 'Main_Tables':mt,'SI_dynamic_tables':[t for t in st if t['table'] not in ('S4','S19','S23')]})
    replacements=[]
    routes={1:'descriptive_summary/data counts',2:'descriptive_summary/fold_counts.csv + edge_coverage.csv',
            3:'diagnostics vocabulary census',6:'descriptive_summary/repeat_metrics.csv',7:'descriptive_summary/repeat_metrics.csv',
            8:'new bootstrap/primary_estimates.csv',9:'descriptive_summary/selection_boundary_summary.csv',
            10:'new diagnostics consistency CSVs',11:'new diagnostics E1 final iteration summary',12:'new diagnostics raw/effective classes',
            13:'new diagnostics internal group census',14:'new diagnostics calibrated loss contributions',
            15:'descriptive_summary/source support',16:'descriptive_summary/source_metrics.csv',17:'descriptive_summary/source_metrics.csv',
            18:'diagnostics/METADATA_CONTEXT_READOUT.json + corrected recurrence CSVs',
            20:'descriptive_summary/external_metrics.csv',21:'new diagnostics external classes and output zeros',
            22:'new diagnostics external ensemble loss',24:'ripk1_case/ensemble_metrics.csv',25:'ripk1_case/contrasts.csv',
            26:'R6_S26_support.csv',27:'R6_S27_contrasts.csv',28:'R6_S28_summary.csv',29:'R6_S29_sensitivity.csv'}
    for table in st:
        n=int(table['table'][1:])
        if n in routes:
            replacements.append({'artifact':'SI','location':'Table '+table['table'],'start_line':table['heading_line'],
                                 'last_data_line':table['last_row_line'],'old_source':str(SI),'replacement_source':routes[n]})
    for table in mt:
        replacements.append({'artifact':'Main','location':'Table '+table['table'],'start_line':table['heading_line'],
                             'last_data_line':table['last_row_line'],'old_source':str(MAIN),
                             'replacement_source':{'1':'descriptive_summary/learner_metrics.csv','2':'new bootstrap/primary_estimates.csv',
                                                   '3':'R6_Table3.csv','4':'ripk1_case/contrasts.csv'}[table['table']]})
    writec('replacement_locations.csv',replacements)
    s31=[]
    def add(key,old,line,source,new_source,unit='',note=''):
        s31.append({'comparison':key,'old_value':old,'new_value':'','unit':unit,'old_source':str(source),
                    'old_line':line,'new_source':new_source,'note':note,'new_status':'WAIT_FOR_STAGE1_ACCEPTANCE'})
    for key,old,line,new,unit in [
        ('compatible_pairs',5788,29,'descriptive_summary/SUMMARY.json','pairs'),
        ('internal_test_observations',3438,31,'descriptive_summary/SUMMARY.json','observations'),
        ('internal_unique_test_pairs',2817,31,'descriptive_summary/SUMMARY.json','pairs'),
        ('binary_E1_better_than_E0matched_repeat_cells','9/9',62,'descriptive_summary/primary_repeat_ratios.csv','cells'),
        ('count_E1_better_than_E0matched_repeat_cells','6/9',62,'descriptive_summary/primary_repeat_ratios.csv','cells'),
        ('count_lower_MAE_than_binary','7/14',84,'descriptive_summary/representation_learner_comparisons.csv','comparisons'),
        ('internal_persistent_zero_observations',19,92,'new diagnostics/internal group census','observations'),
        ('internal_restored_observations',71,92,'new diagnostics/internal group census','observations'),
        ('internal_both_nonzero_observations',3348,92,'new diagnostics/internal group census','observations'),
        ('all_record_binary_input_classes',439,90,'new diagnostics/raw_representation_classes.csv','classes'),
        ('all_record_count_input_classes',493,90,'new diagnostics/raw_representation_classes.csv','classes'),
        ('all_edge_binary_zero_differences',135,90,'new diagnostics/raw_main_pair_diagnostics.csv','pairs'),
        ('all_edge_count_zero_differences',26,90,'new diagnostics/raw_main_pair_diagnostics.csv','pairs'),
        ('bootstrap_contributing_components',38,82,'new bootstrap/primary_estimates.csv','components'),
        ('incomplete_nonsingleton_blocks','19/74',1271,'diagnostics/METADATA_CONTEXT_READOUT.json','blocks'),
        ('recurring_identity_POI_record_comparisons',121,893,'diagnostics/METADATA_CONTEXT_READOUT.json','comparisons'),
        ('recurring_entity_POI_pair_comparisons',556,893,'diagnostics/METADATA_CONTEXT_READOUT.json','comparisons'),
        ('recurring_pair_different_observed_condition',520,900,'diagnostics/METADATA_CONTEXT_READOUT.json','comparisons'),
        ('all60layer_vocab_range','2204-2743',57,'diagnostics/METADATA_CONTEXT_READOUT.json','features'),
        ('outer15layer_vocab_range','2600-2743',57,'diagnostics/METADATA_CONTEXT_READOUT.json','features'),
        ('BRD4_count_lower_MAE','2/14',1389,'descriptive_summary/external_metrics.csv','comparisons'),
        ('BRD4_count_higher_strict_direction','13/14',1389,'descriptive_summary/external_metrics.csv','comparisons'),
        ('BRD4_count_higher_halfcredit_direction','7/14',1389,'descriptive_summary/external_metrics.csv','comparisons')]:
        metadata_key=key in {'incomplete_nonsingleton_blocks','recurring_identity_POI_record_comparisons','recurring_entity_POI_pair_comparisons','recurring_pair_different_observed_condition','all60layer_vocab_range','outer15layer_vocab_range'}
        add(key,old,line,SI if line>1000 or metadata_key else MAIN,new,unit)
    for table in mt:
        for index,row in enumerate(table['rows']):
            for col,value in zip(table['columns'][2:] if table['table'] in ('1','3','4') else table['columns'][1:],
                                 row[2:] if table['table'] in ('1','3','4') else row[1:]):
                add('Table'+table['table']+'|'+'|'.join(row[:2] if table['table'] in ('1','3','4') else row[:1])+'|'+col,
                    value,table['first_row_line']+2+index,MAIN,
                    next(r['replacement_source'] for r in replacements if r['artifact']=='Main' and r['location']=='Table '+table['table']),
                    'dimensionless log ratio/conditional interval' if table['table']=='2' else 'pDC50')
    for row in r6('PRIMARY_SUMMARY.csv','historical'):
        for field in ('theta_E1_minus_E0matched','spearman_support_delta','groups_E1_better','groups_E0matched_better','groups_tied'):
            add('LOCO|'+row['representation']+'|'+field,row[field],102,MAIN,'R6/results/PRIMARY_SUMMARY.csv; scenario=corrected_shadow',
                'pDC50' if field.startswith('theta') else 'correlation/count','Unrounded historical source; Main/SI display rounded')
    writec('S31_old_new_source_map.csv',s31)
    writej('EXTRACTION_RECEIPT.json',{'status':'POINTER_AND_TABLE_EXTRACTION_COMPLETE',
           'source_SHA_checked':{str(k):v for k,v in expected.items()},'source_repair_identities':7,'source_repair_records':12,
           'R6_scenario':'corrected_shadow','R6_acceptance_reused':str(CAND/'r6_budget_b_20261012/ROOT_STAGE_ACCEPTANCE.json'),
           'source_acceptance_reused':str(CAND/'CANDIDATE_ACCEPTANCE.json'),
           'statistical_audit_performed':False,'model_loads':0,'fits':0,'new_predictions':0,
           'R6_reanalysis_performed':False,'S31_new_values':'intentionally blank pending consolidated stage-one acceptance',
           'baseline_table_count':len(mt)+len(st),'replacement_tables':len(replacements),
           'output_line_endings':'UTF-8 without BOM; LF','final_acceptance_owner':'root'})
    print(json.dumps({'S30_identities':7,'S30_records':12,'R6_points':len(points),'S31_rows':len(s31),'replacement_tables':len(replacements)}))


if __name__ == '__main__':
    main()
