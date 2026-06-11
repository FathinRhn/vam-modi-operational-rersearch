# Transportation Problem Solver - VAM & MODI

Aplikasi web Streamlit untuk menyelesaikan Transportation Problem. Solusi awal dihitung dengan Vogel Approximation Method (VAM), lalu dioptimasi dengan Modified Distribution Method (MODI).

## Cara Install

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

```bash
streamlit run app.py
```

## VAM

Vogel Approximation Method menghitung penalty pada setiap baris dan kolom, memilih penalty terbesar, lalu mengalokasikan sebanyak mungkin pada cell biaya terendah. Hasilnya dipakai sebagai solusi awal.

## MODI

Modified Distribution Method menghitung nilai potensial `u` dan `v`, lalu mengevaluasi opportunity cost setiap cell non-basis. Jika semua opportunity cost bernilai lebih besar atau sama dengan nol, solusi sudah optimal. Jika masih ada nilai negatif, alokasi diperbaiki melalui closed loop.

## Struktur Folder

```text
transportation-app/
|-- app.py
|-- requirements.txt
|-- methods/
|   |-- __init__.py
|   |-- vam.py
|   `-- modi.py
|-- utils/
|   |-- __init__.py
|   |-- validation.py
|   `-- formatter.py
`-- README.md
```

## Contoh Input

Supply:

| Nama Sumber | Supply |
| --- | ---: |
| SPPG A | 80 |
| SPPG B | 120 |
| SPPG C | 100 |

Demand:

| Nama Tujuan | Demand |
| --- | ---: |
| SD Negeri 1 | 70 |
| SD Negeri 2 | 90 |
| SMP Negeri 1 | 60 |
| SMA Negeri 1 | 80 |

Matriks biaya:

|  | SD Negeri 1 | SD Negeri 2 | SMP Negeri 1 | SMA Negeri 1 |
| --- | ---: | ---: | ---: | ---: |
| SPPG A | 4 | 6 | 8 | 13 |
| SPPG B | 5 | 11 | 9 | 7 |
| SPPG C | 8 | 7 | 4 | 6 |
