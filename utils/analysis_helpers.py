import pandas as pd
import numpy as np
from itertools import chain, combinations


def TPR_diff(predicted, true, group_info, advantaged, disadvantaged):
    adv_tpr = np.mean(predicted[(true == 1) & (group_info == advantaged)])
    disadv_tpr = np.mean(predicted[(true == 1) & (group_info == disadvantaged)])
    return adv_tpr, disadv_tpr, adv_tpr - disadv_tpr


def DisparateImpact(predicted, true, group_info, advantaged, disadvantaged):
    adv_prob = np.mean(predicted[group_info == advantaged])
    disadv_prob = np.mean(predicted[group_info == disadvantaged])
    return adv_prob, disadv_prob, disadv_prob/adv_prob


def BaseRates(predicted, true, group_info, advantaged, disadvantaged):
    adv_prob = np.mean(true[group_info == advantaged])
    disadv_prob = np.mean(true[group_info == disadvantaged])
    return adv_prob, disadv_prob, disadv_prob/adv_prob


def StatisticalParity_diff(predicted, true, group_info, advantaged, disadvantaged):
    adv_base, disadv_base, __ = BaseRates(predicted, true, group_info, advantaged, disadvantaged)
    adv_out = np.mean(predicted[group_info == advantaged])
    disadv_out = np.mean(predicted[group_info == disadvantaged])
    SP_adv = adv_out/adv_base
    SP_disadv = disadv_out/disadv_base
    return SP_adv, SP_disadv, SP_adv - SP_disadv


def Accuracy_diff(predicted, true, group_info, advantaged, disadvantaged):
    adv_acc = np.mean([predicted[group_info == advantaged] == true[group_info == advantaged]])
    disadv_acc = np.mean([predicted[group_info == disadvantaged] == true[group_info == disadvantaged]])
    return adv_acc, disadv_acc, adv_acc - disadv_acc


def Accuracy_overall(predicted, true):
    return np.mean([predicted == true])


def generate_powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def load_groups_of_interest(group_info, X_test_df):
    test_groups = {}
    # Binary groups
    for group_type in group_info.keys():
        res = {}
        raw_values = X_test_df[group_info[group_type]['column_name']].values
        
        if group_info[group_type]['preprocess'] == 0:
            res['values'] = raw_values
        else:
            # Need to preprocess/threshold:
            threshold = group_info[group_type]['threshold']
            thr_values = np.array([int(x>=int(threshold)) for x in raw_values])
            res['values'] = thr_values
        
        res['advantaged'] = group_info[group_type]['advantaged']
        res['disadvantaged'] = group_info[group_type]['disadvantaged']

        test_groups[group_type] = res

    # Intersectional groups
    all_groups = generate_powerset(group_info.keys())
    intersectional_groups = [x for x in list(all_groups) if len(x)==2]
    for g in intersectional_groups:
        res = {}
        group_name = "_".join(g)
        temp = [str(test_groups[g[0]]['values'][i]) + '_' + str(test_groups[g[1]]['values'][i]) for i in range(len(X_test_df))]
        res['values'] = np.array(temp)
        res['advantaged'] = str(test_groups[g[0]]['advantaged'])+'_'+str(test_groups[g[1]]['advantaged'])
        res['disadvantaged'] = str(test_groups[g[0]]['disadvantaged'])+'_'+str(test_groups[g[1]]['disadvantaged'])

        test_groups[group_name] = res
    
    return test_groups
