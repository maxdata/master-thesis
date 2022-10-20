from information_extraction.analysis.tables import get_wandb_tables, print_latex_table, aggregate_tables
from information_extraction.analysis.stats import perform_aggregation_and_significance_tests
from information_extraction.evaluation import get_evaluator

SWEEP_ID = 'lk072pwe'


def print_representation_overview(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, run_name_parts=[1])
    df_agg.index.name = 'Run'

    significances = perform_aggregation_and_significance_tests(evaluator, tables, baselines=['text', 'html_base'],
                                                               p_value=0.01, run_name_parts=[1])

    print(significances)

    print_latex_table(df_agg.style.format_index('\\verb|{}|'),
                      caption='Performance of our extraction models, using different representations. '
                              'Best results are marked in bold.',
                      label='table:representation_performance', highlight_axis='index')


def main():
    tables = get_wandb_tables(SWEEP_ID)

    tables = {
        run_name.split('-')[0] + '-' + '_'.join(run_name.split('-')[1:]): table
        for run_name, table in tables.items()
    }

    evaluator = get_evaluator()
    print_representation_overview(evaluator, tables)


if __name__ == '__main__':
    main()
