import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

# Global dataframe
df = None

def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            df = pd.read_csv(file_path)
            show_table(df)
            stats_output.delete("1.0", tk.END)
            stats_output.insert(tk.END, "✅ CSV loaded successfully.\n")
            messagebox.showinfo("Success", "CSV loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file:\n{e}")

def show_table(data):
    for row in tree.get_children():
        tree.delete(row)

    tree["column"] = list(data.columns)
    tree["show"] = "headings"

    for col in data.columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)

    for i, row in data.head(50).iterrows():
        tree.insert("", "end", values=list(row))

def show_stats():
    if df is not None:
        try:
            numeric_cols = df.select_dtypes(include=['float64', 'int64'])

            stats_output.delete("1.0", tk.END)
            stats_output.insert(tk.END, "📊 Descriptive Statistics (Numeric Columns):\n\n")

            if numeric_cols.empty:
                stats_output.insert(tk.END, "No numeric columns found in the dataset.\n")
            else:
                desc = numeric_cols.describe().T  # Transpose for better readability
                variance = numeric_cols.var()

                for col in desc.index:
                    stats_output.insert(tk.END, f"📌 {col}:\n")
                    stats_output.insert(tk.END, f"   • Count    : {desc.loc[col, 'count']:.0f}\n")
                    stats_output.insert(tk.END, f"   • Mean     : {desc.loc[col, 'mean']:.2f}\n")
                    stats_output.insert(tk.END, f"   • Std Dev  : {desc.loc[col, 'std']:.2f}\n")
                    stats_output.insert(tk.END, f"   • Variance : {variance[col]:.2f}\n")
                    stats_output.insert(tk.END, f"   • Min      : {desc.loc[col, 'min']:.2f}\n")
                    stats_output.insert(tk.END, f"   • Median   : {desc.loc[col, '50%']:.2f}\n")
                    stats_output.insert(tk.END, f"   • Max      : {desc.loc[col, 'max']:.2f}\n\n")

            # Additional categorical info
            total_missions = len(df)
            companies = df['Company Name'].nunique() if 'Company Name' in df else "N/A"
            successes = df['Status Mission'].str.contains('Success', case=False).sum()
            failures = df['Status Mission'].str.contains('Failure', case=False).sum()

            stats_output.insert(tk.END, "📌 General Mission Summary:\n")
            stats_output.insert(tk.END, f"   • Total Missions     : {total_missions}\n")
            stats_output.insert(tk.END, f"   • Unique Companies   : {companies}\n")
            stats_output.insert(tk.END, f"   • Successful Missions: {successes}\n")
            stats_output.insert(tk.END, f"   • Failed Missions    : {failures}\n")

        except Exception as e:
            stats_output.insert(tk.END, f"❌ Error calculating stats: {e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")


def show_probability():
    if df is not None:
        try:
            total = len(df)
            success = df['Status Mission'].str.contains('success', case=False).sum()
            failure = df['Status Mission'].str.contains('failure', case=False).sum()

            prob_success = success / total
            prob_failure = failure / total

            stats_output.delete("1.0", tk.END)
            stats_output.insert(tk.END, "📈 Probability Analysis:\n")
            stats_output.insert(tk.END, f"• Probability of Success: {prob_success:.2f}\n")
            stats_output.insert(tk.END, f"• Probability of Failure: {prob_failure:.2f}\n")
        except Exception as e:
            stats_output.insert(tk.END, f"❌ Error in probability calculation: {e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")

def plot_company_pie():
    if df is not None:
        try:
            company_counts = df['Company Name'].value_counts().head(10)
            plt.figure(figsize=(8, 6))
            plt.pie(company_counts, labels=company_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title("Top 10 Launches by Company")
            plt.axis("equal")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting pie chart:\n{e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")

def plot_missions_by_year():
    if df is not None:
        try:
            df['Year'] = pd.to_datetime(df['Datum'], errors='coerce').dt.year
            year_counts = df['Year'].value_counts().sort_index()
            if not year_counts.empty:
                plt.figure(figsize=(10, 5))
                plt.plot(year_counts.index, year_counts.values, marker='o', color='blue')
                plt.title("Missions Per Year")
                plt.xlabel("Year")
                plt.ylabel("Number of Missions")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                messagebox.showinfo("Info", "No valid 'Datum' data to extract years.")
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting missions by year:\n{e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")

def plot_success_failure():
    if df is not None:
        try:
            status_counts = df['Status Mission'].str.lower().value_counts()
            color_map = {'success': 'green', 'failure': 'red', 'partial failure': 'orange'}
            colors = [color_map.get(status, 'gray') for status in status_counts.index]

            plt.figure(figsize=(6, 4))
            status_counts.plot(kind='bar', color=colors)
            plt.title("Mission Outcome Counts")
            plt.xlabel("Status")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Error plotting status bar chart:\n{e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")

def show_regression():
    if df is not None:
        try:
            df['Year'] = pd.to_datetime(df['Datum'], errors='coerce').dt.year
            year_counts = df['Year'].value_counts().sort_index()
            X = np.array(year_counts.index).reshape(-1, 1)
            y = np.array(year_counts.values).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)

            # 95% Confidence Interval
            n = len(X)
            mean_x = np.mean(X)
            t_score = stats.t.ppf(0.975, df=n - 1)
            se = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2)) / np.sqrt(np.sum((X - mean_x) ** 2))
            ci_margin = t_score * se
            upper = y_pred + ci_margin
            lower = y_pred - ci_margin

            plt.figure(figsize=(10, 5))
            plt.scatter(X, y, color='blue', label='Actual')
            plt.plot(X, y_pred, color='red', label='Predicted')
            plt.fill_between(X.flatten(), lower.flatten(), upper.flatten(), color='orange', alpha=0.3, label='95% Confidence Interval')
            plt.xlabel("Year")
            plt.ylabel("Number of Missions")
            plt.title("Linear Regression with 95% Confidence Interval")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Error", f"Regression failed:\n{e}")
    else:
        messagebox.showwarning("No Data", "Please load a CSV file first.")

# ---------------- GUI SETUP ---------------- #
root = tk.Tk()
root.title("🚀 Space Mission Analyzer")
root.geometry("1050x800")
root.configure(bg="#f0f2f5")

title_font = ("Segoe UI", 18, "bold")
button_font = ("Segoe UI", 11, "bold")

label = tk.Label(root, text="🚀 Space Mission Analyzer", font=title_font, bg="#f0f2f5", fg="#1f1f2e")
label.pack(pady=10)

# Buttons
button_frame = tk.Frame(root, bg="#f0f2f5")
button_frame.pack(pady=10)

tk.Button(button_frame, text="📂 Load CSV", command=load_csv, font=button_font,
          bg="#007acc", fg="white", padx=10, pady=5).grid(row=0, column=0, padx=10)

tk.Button(button_frame, text="📊 Show Statistics", command=show_stats, font=button_font,
          bg="#28a745", fg="white", padx=10, pady=5).grid(row=0, column=1, padx=10)

tk.Button(button_frame, text="🧮 Show Probability", command=show_probability, font=button_font,
          bg="#ff9800", fg="white", padx=10, pady=5).grid(row=0, column=2, padx=10)

# Graph buttons
graph_frame = tk.Frame(root, bg="#f0f2f5")
graph_frame.pack(pady=5)

tk.Button(graph_frame, text="🍕 Pie: Launches by Company", command=plot_company_pie, font=button_font,
          bg="#6f42c1", fg="white", padx=10).grid(row=0, column=0, padx=5, pady=5)

tk.Button(graph_frame, text="📈 Line: Missions by Year", command=plot_missions_by_year, font=button_font,
          bg="#17a2b8", fg="white", padx=10).grid(row=0, column=1, padx=5, pady=5)

tk.Button(graph_frame, text="📉 Bar: Success vs Failure", command=plot_success_failure, font=button_font,
          bg="#dc3545", fg="white", padx=10).grid(row=0, column=2, padx=5, pady=5)

tk.Button(graph_frame, text="🔮 Regression: Missions/Year", command=show_regression, font=button_font,
          bg="#20c997", fg="white", padx=10).grid(row=0, column=3, padx=5, pady=5)

# Output and Table
stats_output = tk.Text(root, height=8, width=100, font=("Courier New", 10))
stats_output.pack(pady=10)

table_frame = tk.Frame(root)
table_frame.pack(fill="both", expand=True)

tree_scroll = tk.Scrollbar(table_frame)
tree_scroll.pack(side="right", fill="y")

tree = ttk.Treeview(table_frame, yscrollcommand=tree_scroll.set)
tree.pack(fill="both", expand=True)
tree_scroll.config(command=tree.yview)

root.mainloop()