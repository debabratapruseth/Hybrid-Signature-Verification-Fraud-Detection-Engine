# Private case data

The consolidated Colab runner creates a separate private case folder:

```text
data/
└── cases/
    └── <run-id>/
        ├── questioned/
        │   └── questioned_document.pdf
        └── references/
            ├── reference_01.png
            ├── reference_02.png
            └── reference_03.png
```

Case data is ignored by Git. Use only authorized material, keep the original
questioned file unchanged, and do not commit signatures or identifying data.

The three references must be separately produced samples from the same claimed
writer and should use the same normal signature form as the questioned sample.
