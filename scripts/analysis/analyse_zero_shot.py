from information_extraction.analysis.tables import get_wandb_tables, print_latex_table, aggregate_tables
from information_extraction.evaluation import get_evaluator

SWEEP_ID = 'chxe9bvx'


def print_zero_shot_performance(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, run_name_parts=[2,3]).groupby(level=0).mean().mean().to_frame('Ours').T

    print_latex_table(df_agg, caption='Performance of our extraction models in the zero-shot setting, compared to two '
                                      'strong baselines. Best performances are marked in bold.',
                      label='table:zero_shot_performance')


def print_zero_shot_per_attribute(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, per_vertical=True, per_attribute=True,
                              run_name_parts=[3])
    df_agg = df_agg.groupby(level=[1, 2]).mean()

    table = df_agg.style.format_index(lambda s: 'NBA player' if s == 'nbaplayer' else s.capitalize(), level=0)
    table = table.format_index('\\verb|{}|', level=1)

    print_latex_table(table, caption='Performance of our extraction models in the zero-shot setting, per attribute.',
                      label='table:zero_shot_per_attribute', long_table=True)


def main():
    tables = get_wandb_tables(SWEEP_ID)

    evaluator = get_evaluator()
    print_zero_shot_performance(evaluator, tables)
    print_zero_shot_per_attribute(evaluator, tables)


if __name__ == '__main__':
    main()
