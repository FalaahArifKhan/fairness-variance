# Overview

## analyzers


Subgroups Statistical Bias and Variance Analyzers.

This module contains fairness and stability analysing methods for defined subgroups.
The purpose of an analyzer is to analyse defined metrics for defined subgroups.


- [AbstractOverallVarianceAnalyzer](../analyzers/AbstractOverallVarianceAnalyzer)
- [AbstractSubgroupsAnalyzer](../analyzers/AbstractSubgroupsAnalyzer)
- [BatchOverallVarianceAnalyzer](../analyzers/BatchOverallVarianceAnalyzer)
- [SubgroupsStatisticalBiasAnalyzer](../analyzers/SubgroupsStatisticalBiasAnalyzer)
- [SubgroupsVarianceAnalyzer](../analyzers/SubgroupsVarianceAnalyzer)
- [SubgroupsVarianceCalculator](../analyzers/SubgroupsVarianceCalculator)

## custom_classes

- [BaseDataset](../custom-classes/BaseDataset)
- [CompasDataset](../custom-classes/CompasDataset)
- [CompasWithoutSensitiveAttrsDataset](../custom-classes/CompasWithoutSensitiveAttrsDataset)
- [GenericPipeline](../custom-classes/GenericPipeline)
- [MetricsComposer](../custom-classes/MetricsComposer)
- [MetricsVisualizer](../custom-classes/MetricsVisualizer)

## metrics

- [compute_churn](../metrics/compute-churn)
- [compute_conf_interval](../metrics/compute-conf-interval)
- [compute_entropy](../metrics/compute-entropy)
- [compute_jitter](../metrics/compute-jitter)
- [compute_per_sample_accuracy](../metrics/compute-per-sample-accuracy)
- [compute_stability_metrics](../metrics/compute-stability-metrics)

## preprocessing


## user_interfaces


User interfaces.

This module contains user interfaces for metrics computation.


- [run_metrics_computation_with_config](../user-interfaces/run-metrics-computation-with-config)

## utils


