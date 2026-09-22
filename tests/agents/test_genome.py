import pytest

from agents import Genome


def test_genome_is_created_with_genes():
    genes = [0, 1, 2, 3]

    genome = Genome(genes)

    assert genome.genes == genes

def test_random_genome_has_requested_length():
    genome = Genome.random(100)

    assert len(genome) == 100

def test_random_genome_contains_only_valid_actions():
    genome = Genome.random(100)

    assert all(gene in {0, 1, 2, 3} for gene in genome.genes)

def test_get_gene_returns_gene_at_index(): 
    genome = Genome([0, 1, 2, 3])

    gene = genome.get_gene(2)

    assert gene == 2

def test_copy_creates_equal_genome(): 
    genome = Genome([0, 1, 2, 3])

    copied_genome = genome.copy()

    assert copied_genome.genes == genome.genes

# def test_genome_rejects_invalid_action():
#     with pytest.raises(ValueError):
#         Genome([0, 1, 4, 3])

# def test_genome_rejects_negative_action():
#     with pytest.raises(ValueError):
#         Genome([0, -1, 2, 3])