from medassist.rag.retriever import format_context, source_label, source_labels


def test_source_label_formats_page_as_1_indexed(fake_docs):
    assert source_label(fake_docs[0]) == "disease_book.pdf p.4"


def test_source_labels_dedupes(fake_docs):
    labels = source_labels(fake_docs + [fake_docs[0]])
    assert labels.count("disease_book.pdf p.4") == 1


def test_format_context_includes_all_chunks(fake_docs):
    ctx = format_context(fake_docs)
    assert "Influenza" in ctx and "Paracetamol" in ctx
    assert "[1]" in ctx and "[2]" in ctx
