# Excel ML App (Flask + scikit-learn)

A beginner-friendly web application that allows users to upload an Excel file (`.xlsx`), train a machine-learning model, and make predictions using the column headers.

This project is designed for **learning, experimentation, and rapid prototyping** with tabular data.

---

## ✨ Features

- Upload Excel `.xlsx` files
- Automatically uses:
  - **All columns except the last** as input features (X)
  - **Last column** as target/output (y)
- Automatically detects:
  - **Regression** (numeric output)
  - **Classification** (categorical output)
- User-selectable models:
  - **MLP (Neural Network)** – epochs + loss curve
  - **Random Forest** – strong baseline for tabular data
- Handles missing values and categorical features
- Clean, modern UI
- Keeps input values after prediction
- Per-session model training (no login required)

---

## 🧠 How It Works

1. Upload an Excel file (`.xlsx`)
2. The app splits:
   - X = all columns except the last
   - y = last column
3. Applies preprocessing:
   - Numeric → median imputation
   - Categorical → one-hot encoding
4. User selects model (MLP or RF)
5. Model is trained and evaluated
6. User enters values to predict output

---

## 📊 Supported Tasks

| Task | Condition |
|-----|----------|
| Regression | Numeric output column |
| Classification | Categorical output column |

---

## 🗂️ Project Structure

```
excel-ml-app/
├─ app.py
├─ config.py
├─ requirements.txt
├─ README.md
├─ .gitignore
│
├─ webapp/
│  ├─ routes/
│  ├─ services/
│  ├─ templates/
│  └─ static/
```

---

## ⚙️ Installation (Local)

```bash
git clone https://github.com/<YOUR_USERNAME>/excel-ml-app.git
cd excel-ml-app

conda create -n excel-ml-app python=3.11 -y
conda activate excel-ml-app

pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000

---

## 📄 License

MIT License

---

## 👤 Author

© 2026 Nazmul
