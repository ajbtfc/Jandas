# Jandas: Pandas-Like DataFrames in Ignition

Welcome to **Jandas**, a lightweight Python library designed to bring familiar `pandas`-style data manipulation to the **Ignition platform**.

Jandas provides a simple, intuitive interface for working with tabular data in environments where the full power of `pandas` isn’t available — such as Jython scripting in Ignition. If you've ever wished for `.groupby()`, `.rolling()`, or `.loc[]` support while working inside your Ignition gateway or Perspective scripts, **Jandas aims to give you just that**.

---

## ⚙️ What Jandas Is

- A minimal re-implementation of core `pandas` features, tailored for environments like **Ignition** where the real `pandas` cannot run.
- Designed to work with Python objects like lists and dictionaries, common in Ignition datasets.
- Offers a familiar interface with classes like `DataFrame`, `Series`, and `GroupBy`.

---

## 🚧 Limitations

- Jandas is written in **pure Python**, not C or NumPy — which means it **does not match the performance** of `pandas`, especially for large datasets.
- It is focused on **developer convenience** and **readability**, not computational efficiency.
- Some advanced features of `pandas` (e.g. time series indexing, multi-indexing, broadcasting, custom dtypes) are **not fully implemented**.
- This project is still **under active development** — interfaces, behavior, and performance may change over time.

---

## 🔍 When to Use Jandas

- You're working inside **Ignition** or another Jython environment without access to `pandas`.
- You want to write **clear, expressive, and testable** data logic that resembles modern Python data workflows.
- You’re operating on **moderate-sized** datasets where performance isn’t the top concern.

---

## 🚫 When *Not* to Use Jandas

- You need **high-performance** data manipulation (use native `pandas` in a proper Python runtime instead).
- Your dataset contains millions of rows and complex statistical transformations.
- You require third-party integration with the broader Python data science ecosystem.

---

## 📦 Project Status

Jandas is **alpha-stage software**. While the core functionality is usable, the API may change, and performance optimizations are ongoing.

You are welcome to use it, experiment with it, and contribute ideas or code — but be prepared for occasional sharp edges!

---

## 📚 Next Steps

- [DataFrame](dataframe_dataframe.md)
- [JIndex](dataframe_jindex.md)
- [GroupBy](dataframe_groupby.md)
- [Rolling](rolling.md)

For Ignition users and curious developers alike, we hope Jandas helps bridge the gap between industrial scripting and modern data analysis.

---