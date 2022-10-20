import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from information_extraction.analysis.tables import get_wandb_tables, aggregate_tables, print_latex_table
from information_extraction.evaluation import get_evaluator
from information_extraction.data.ground_truths import GROUND_TRUTHS
from information_extraction.data.metrics import compute_f1, compute_exact

SWEEP_ID = 'v2h7wsaa'

plt.style.use('seaborn')


def sample_failures(tables, num_failures: int = 20):
    new_rows = []

    for run_name, df in tables.items():
        attributes = [x[:-5] for x in df.columns if x.endswith('/true')]

        for attribute in attributes:
            df_sub = df[df[f'{attribute}/em'] == 0]
            df_sub = df_sub.sample(min(num_failures, len(df_sub)))

            for _, row in df_sub.iterrows():
                vertical, website, doc_id = row['doc_id'].split('/')

                new_rows.append({
                    'vertical': vertical,
                    'website': website,
                    'doc_id': doc_id,
                    'attribute': attribute,
                    'true': row[f'{attribute}/true'],
                    'pred': row[f'{attribute}/pred'],
                })

    df = pd.DataFrame(new_rows)

    df.to_csv('failures.csv', index=False)


def show_failure_types():
    df = pd.read_csv('failures_with_reasons.csv')
    failure_types = df['reason'].value_counts()

    percentages = (failure_types / failure_types.sum() * 100).apply('({:.1f}%)'.format)

    labels = np.where(failure_types / failure_types.sum() < 0.04,
                      failure_types.index + ' ' + percentages,
                      failure_types.index + '\n' + percentages)

    fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]}, figsize=(9, 6))

    wedges, texts = ax.pie(failure_types, labels=labels, textprops={'ha': 'center'})
    lax.legend(wedges, failure_types.index + ' (' + failure_types.astype(str) + ')',
               loc='center right', title='Failure types')
    fig.suptitle('Distribution of failure types')

    texts[-1].set_position((texts[-1]._x, texts[-1]._y + .02))
    texts[-2].set_position((texts[-2]._x, texts[-2]._y - .02))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('failure_type_distribution.png')


def show_different_performances_per_website(evaluator, tables):
    df_agg = aggregate_tables(evaluator, tables, per_website=True, per_attribute=True, per_vertical=True)
    df_agg = df_agg[('Instance', 'F1')]

    fig, axes = plt.subplots(4, 2, sharex='all', figsize=(10, 6))

    for (vertical, df_vertical), ax in zip(df_agg.groupby(level=0), axes.flat):

        websites = sorted(df_vertical.index.get_level_values(1).unique())
        attributes = sorted(df_vertical.index.get_level_values(2).unique())

        data = np.zeros((len(attributes), len(websites)))

        for i, attribute in enumerate(attributes):
            for j, website in enumerate(websites):
                data[i, j] = df_vertical.loc[(vertical, website, attribute)]

        colours = ax.pcolormesh(data, vmin=0, vmax=1, cmap='inferno')

        ax.set_yticks(np.arange(len(attributes)) + 0.5)
        ax.set_yticklabels(attributes)

        ax.set_xticks(np.arange(len(websites)) + 0.5)
        ax.set_xticklabels([''] * len(websites))

        ax.set_title(vertical)

    fig.subplots_adjust(right=0.8, wspace=0.6, hspace=0.4)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(colours, cax=cbar_ax)
    cbar.set_label('Instance-level WM score')

    fig.suptitle('Differences in performance per website')
    plt.savefig('performance_per_website.png')


def compute_mean_reciprocal_ranks(tables):
    mrr_rows = []
    ranks = []
    for run_name, df in tables.items():
        vertical = run_name.split('-')[-1]

        for attribute, df_attr in df.groupby('feature'):
            reciprocal_ranks = []

            for doc_id, df_sub in df_attr.groupby('doc_id'):
                if GROUND_TRUTHS[doc_id][attribute] == ['']:
                    # We can't compute the rank if the value does not exist on the page
                    continue

                df_sub['expected'] = [
                    max(GROUND_TRUTHS[doc_id][attribute], key=lambda a_gold: compute_f1(a_gold, a_pred))
                    for a_pred in df_sub['predicted']
                ]
                df_sub['f1'] = [
                    compute_f1(a_gold, a_pred)
                    for a_gold, a_pred in zip(df_sub['expected'], df_sub['predicted'])
                ]
                df_sub['em'] = [
                    compute_exact(a_gold, a_pred)
                    for a_gold, a_pred in zip(df_sub['expected'], df_sub['predicted'])
                ]

                df_sub = df_sub[df_sub['predicted'] != ''].sort_values('score', ascending=False)

                # If no found value is correct, we also cannot compute the rank.
                # Note that we deviate from the standard MRR where no correct prediction implies a reciprocal rank of 0.
                # The reason for doing so: we are only interested in failures of the reranking stage, not failures of
                # the extraction approach as a whole.
                if df_sub['f1'].max() > 0:
                    rank = df_sub['f1'].argmax() + 1

                    ranks.append(rank)
                    reciprocal_ranks.append(1 / rank)

            mrr_rows.append({
                'vertical': vertical,
                'attribute': attribute,
                'mrr': np.mean(reciprocal_ranks),
            })

    df = pd.DataFrame(mrr_rows)

    return ranks, df


def print_mrrs(df):
    df.rename(columns={'vertical': 'Vertical', 'attribute': 'Attribute', 'mrr': 'MRR'}, inplace=True)
    df.set_index(['Vertical', 'Attribute'], inplace=True)
    df.sort_index(inplace=True)

    table = df.style.format_index(lambda s: 'NBA player' if s == 'nbaplayer' else s.capitalize(), level=0)
    table = table.format_index('\\verb|{}|', level=1)

    print_latex_table(table, 'Attribute-level mean reciprocal rank (MRR) of correct segment predictions.',
                      'table:attribute_level_mrr', long_table=True)


def show_rank_distribution(ranks):
    counts = [0 for _ in range(max(ranks))]

    for rank in ranks:
        counts[rank - 1] += 1

    labels = list(range(1, len(counts) + 1))

    fig, ax = plt.subplots(1, 1)
    ax.bar(labels, counts)

    ax.set_xticks(np.array(labels))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Rank of segment with best prediction')
    ax.set_ylabel('Number of queries (document + attribute)')
    ax.set_yscale('log')

    fig.suptitle('Rank distribution of the best extraction results')

    plt.savefig('rank_distribution.png')


def main():
    document_tables = {
        key: value
        for key, value in get_wandb_tables(SWEEP_ID).items()
        if 'bert' in key
    }
    segment_tables = {
        key: value
        for key, value in get_wandb_tables(SWEEP_ID, table_type='segments').items()
        if 'bert' in key
    }
    evaluator = get_evaluator()

    sample_failures(document_tables)
    show_different_performances_per_website(evaluator, document_tables)

    show_failure_types()

    ranks, df_mrr = compute_mean_reciprocal_ranks(segment_tables)
    show_rank_distribution(ranks)
    print_mrrs(df_mrr)


if __name__ == '__main__':
    main()
