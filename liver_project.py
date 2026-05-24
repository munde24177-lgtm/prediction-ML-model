#Liver disease -----------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import os
os.makedirs('plots', exist_ok=True)


# PALETTE & THEME SETUP--------------------------------------------------------------------------------

PALETTE = {
    'bg':        '#0D1117',
    'surface':   '#161B22',
    'surface2':  '#1C2333',
    'border':    '#30363D',
    'accent1':   '#58A6FF',   
    'accent2':   '#3FB950',   
    'accent3':   '#FF7B72',   
    'accent4':   '#FFA657',   
    'accent5':   '#D2A8FF',   
    'text':      '#E6EDF3',
    'subtext':   '#8B949E',
}

CLASS_COLORS = {
    'Healthy Liver':                  '#3FB950',
    'Fatty Liver Disease (NAFLD)':    '#FFA657',
    'Alcoholic Liver Disease':        '#FF7B72',
    'General Liver Disease Severity': '#58A6FF',
    'Liver Cirrhosis Risk':           '#D2A8FF',
}

def apply_dark_theme():
    plt.rcParams.update({
        'figure.facecolor':   PALETTE['bg'],
        'axes.facecolor':     PALETTE['surface'],
        'axes.edgecolor':     PALETTE['border'],
        'axes.labelcolor':    PALETTE['text'],
        'text.color':         PALETTE['text'],
        'xtick.color':        PALETTE['subtext'],
        'ytick.color':        PALETTE['subtext'],
        'grid.color':         PALETTE['border'],
        'grid.alpha':         0.5,
        'legend.facecolor':   PALETTE['surface2'],
        'legend.edgecolor':   PALETTE['border'],
        'font.family':        'DejaVu Sans',
    })

apply_dark_theme()

def save(name):
    path = f'plots/{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=PALETTE['bg'], edgecolor='none')
    plt.close('all')
    print(f'  ✓ Saved → {name}.png')

def section(title):
    print(f'\n{"─"*70}')
    print(f'  {title}')
    print(f'{"─"*70}')


# 1. LOAD DATA---------------------------------------------------------------------------------------------

section('1 / LOADING DATASET')
df_raw = pd.read_csv("Training_Liver_Disease_Dataset.csv")
print(f'  Rows: {df_raw.shape[0]}  │  Columns: {df_raw.shape[1]}')
print(f'  Target classes: {df_raw["Liver_Disease_Class"].unique().tolist()}')


# 2. DATA PREPROCESSING------------------------------------------------------------------------------------

section('2 / DATA PREPROCESSING')
df = df_raw.copy()

# Drop columns with no predictive value
df.drop(columns=['Source'], inplace=True)

# 2a. Handling Missing Values-------------------------------------------------------------------------------
print('  Missing values before:')
print(df.isnull().sum()[df.isnull().sum() > 0].to_string())

# Alcohol_Consumption: fill with 'None' 
df['Alcohol_Consumption'] = df['Alcohol_Consumption'].fillna('None')
# Medication_History: fill with 'None'
df['Medication_History'] = df['Medication_History'].fillna('None')

# Numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

print(f'\n  Missing values after: {df.isnull().sum().sum()}  ✓')

# 2b. Encode Category----------------------------------------------------------------------------------
ordered_maps = {
    'Diet_Quality':       {'Poor': 0, 'Average': 1, 'Healthy': 2},
    'Physical_Activity':  {'Sedentary': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    'Obesity_Class':      {'Underweight': 0, 'Normal': 1, 'Overweight': 2,
                           'Obesity I': 3, 'Obesity II': 4, 'Obesity III': 5},
    'Alcohol_Consumption':{'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
    'Smoking_Status':     {'Never': 0, 'Former': 1, 'Current': 2},
}
for col, mapping in ordered_maps.items():
    df[col] = df[col].map(mapping).fillna(0).astype(int)

# Binary encoding
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Label-encoding ---------------------------------------------------------------------
le = LabelEncoder()
for col in ['Occupation', 'Medication_History']:
    df[col] = le.fit_transform(df[col].astype(str))

# Target encoding
target_le = LabelEncoder()
df['target'] = target_le.fit_transform(df['Liver_Disease_Class'])
class_names = target_le.classes_

# 2c.  Target split-------------------------------------------------------------------------------
drop_cols = ['Liver_Disease_Class', 'target']
X = df.drop(columns=drop_cols)
y = df['target']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test 
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)
print(f'\n  Train size: {len(X_train)}  │  Test size: {len(X_test)}')
print(f'  Features used: {X.shape[1]}')


# 3. EDA----------------------------------------------------------------------------------------------

section('3 / EXPLORATORY DATA ANALYSIS')

# Plot 1: Class Distribution------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('CLASS DISTRIBUTION', fontsize=16, fontweight='bold',
             color=PALETTE['text'], y=1.02)

counts = df_raw['Liver_Disease_Class'].value_counts()
colors_list = [CLASS_COLORS[c] for c in counts.index]

# Bar chart------------------------------------------------------------------------------------------------
ax = axes[0]
bars = ax.barh(counts.index, counts.values, color=colors_list,
               edgecolor=PALETTE['bg'], linewidth=1.5)
ax.set_facecolor(PALETTE['surface'])
ax.set_xlabel('Number of Patients', color=PALETTE['text'], fontsize=11)
ax.set_title('Patient Count by Disease Class', color=PALETTE['text'],
             fontsize=12, pad=10)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2,
            f'{val:,}', va='center', ha='left',
            color=PALETTE['subtext'], fontsize=10)
ax.spines[:].set_color(PALETTE['border'])
ax.tick_params(colors=PALETTE['subtext'])
ax.grid(axis='x', alpha=0.3, color=PALETTE['border'])

# Pie chart------------------------------------------------------------------------------------------------
ax2 = axes[1]
ax2.set_facecolor(PALETTE['surface'])
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=None, colors=colors_list,
    autopct='%1.1f%%', pctdistance=0.80,
    wedgeprops={'edgecolor': PALETTE['bg'], 'linewidth': 2})
for at in autotexts:
    at.set_color(PALETTE['bg'])
    at.set_fontsize(9)
    at.set_fontweight('bold')
ax2.legend(wedges, counts.index, title='Disease Classes',
           loc='lower center', bbox_to_anchor=(0.5, -0.20),
           frameon=True, fontsize=8, ncol=2,
           facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])
ax2.set_title('Proportion of Each Class', color=PALETTE['text'],
              fontsize=12, pad=10)

plt.tight_layout()
save('01_class_distribution')

# ── Plot 2 Enzymes vs Disease ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('LIVER ENZYME LEVELS BY DISEASE CLASS',
             fontsize=16, fontweight='bold', color=PALETTE['text'])

for ax, col, unit, clr in zip(axes,
                               ['ALT', 'AST', 'Bilirubin'],
                               ['U/L', 'U/L', 'mg/dL'],
                               [PALETTE['accent1'], PALETTE['accent3'], PALETTE['accent4']]):
    order = list(CLASS_COLORS.keys())
    clr_list = list(CLASS_COLORS.values())
    data_by_class = [df_raw[df_raw['Liver_Disease_Class'] == c][col].dropna()
                     for c in order]
    vp = ax.violinplot(data_by_class, positions=range(len(order)),
                       showmedians=True, showextrema=True)
    for i, (body, c) in enumerate(zip(vp['bodies'], clr_list)):
        body.set_facecolor(c)
        body.set_alpha(0.7)
        body.set_edgecolor(PALETTE['border'])
    vp['cmedians'].set_color(PALETTE['text'])
    vp['cmedians'].set_linewidth(2)
    for part in ['cbars', 'cmaxes', 'cmins']:
        vp[part].set_color(PALETTE['border'])

    ax.set_facecolor(PALETTE['surface'])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.replace(' ', '\n') for c in order], fontsize=7,
                       color=PALETTE['subtext'])
    ax.set_ylabel(unit, color=PALETTE['text'], fontsize=10)
    ax.set_title(col, color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.spines[:].set_color(PALETTE['border'])
    ax.tick_params(colors=PALETTE['subtext'])
    ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])

plt.tight_layout()
save('02_enzymes_by_class')

# Plot 3: Alco & smoke vs Disease---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('LIFESTYLE FACTORS VS DISEASE CLASS',
             fontsize=16, fontweight='bold', color=PALETTE['text'])

for ax, col, title in zip(axes,
                          ['Alcohol_Consumption', 'Smoking_Status'],
                          ['Alcohol Consumption', 'Smoking Status']):
    ct = pd.crosstab(df_raw[col], df_raw['Liver_Disease_Class'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    c_list = [CLASS_COLORS.get(c, PALETTE['accent1']) for c in ct_pct.columns]
    ct_pct.plot(kind='bar', ax=ax, color=c_list, width=0.75,
                edgecolor=PALETTE['bg'], linewidth=0.8)
    ax.set_facecolor(PALETTE['surface'])
    ax.set_title(title, color=PALETTE['text'], fontsize=13, fontweight='bold')
    ax.set_ylabel('Proportion (%)', color=PALETTE['text'], fontsize=10)
    ax.set_xlabel('', color=PALETTE['text'])
    ax.tick_params(axis='x', rotation=30, colors=PALETTE['subtext'])
    ax.tick_params(axis='y', colors=PALETTE['subtext'])
    ax.spines[:].set_color(PALETTE['border'])
    ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])
    ax.legend(title='Disease', fontsize=7, title_fontsize=8,
              facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])

plt.tight_layout()
save('03_lifestyle_factors')

# Plot 4: BMI  --------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(PALETTE['bg'])
ax.set_facecolor(PALETTE['surface'])

for cls, clr in CLASS_COLORS.items():
    subset = df_raw[df_raw['Liver_Disease_Class'] == cls]['BMI'].dropna()
    ax.hist(subset, bins=25, alpha=0.55, color=clr, label=cls,
            edgecolor=PALETTE['bg'], linewidth=0.5)

ax.set_xlabel('BMI', color=PALETTE['text'], fontsize=12)
ax.set_ylabel('Count', color=PALETTE['text'], fontsize=12)
ax.set_title('BMI DISTRIBUTION BY DISEASE CLASS',
             color=PALETTE['text'], fontsize=15, fontweight='bold')
ax.spines[:].set_color(PALETTE['border'])
ax.tick_params(colors=PALETTE['subtext'])
ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])
ax.legend(fontsize=9, facecolor=PALETTE['surface2'],
          labelcolor=PALETTE['text'])
plt.tight_layout()
save('04_bmi_distribution')

# Plot 5: Correlation Heatmap -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 13))
fig.patch.set_facecolor(PALETTE['bg'])
ax.set_facecolor(PALETTE['surface'])

num_features = ['Age', 'BMI', 'Waist_Circumference', 'Sleep_Hours',
                'ALT', 'AST', 'Bilirubin', 'Albumin', 'Platelets',
                'Alk_Phosphatase', 'GGT', 'Triglycerides', 'INR',
                'Alcohol_Consumption', 'Smoking_Status', 'target']
corr = df[num_features].corr()

cmap = LinearSegmentedColormap.from_list(
    'custom', [PALETTE['accent3'], PALETTE['bg'], PALETTE['accent1']])
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, ax=ax, mask=mask, cmap=cmap, center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7, 'color': PALETTE['text']},
            linewidths=0.3, linecolor=PALETTE['border'],
            cbar_kws={'shrink': 0.8})

ax.set_facecolor(PALETTE['surface'])
ax.set_title('FEATURE CORRELATION MATRIX',
             color=PALETTE['text'], fontsize=15, fontweight='bold', pad=15)
ax.tick_params(colors=PALETTE['subtext'], labelsize=8)
plt.xticks(rotation=45, ha='right', color=PALETTE['subtext'])
plt.yticks(rotation=0, color=PALETTE['subtext'])
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(colors=PALETTE['subtext'])

plt.tight_layout()
save('05_correlation_heatmap')

#   Age & Gender distribution -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('AGE & GENDER BREAKDOWN BY DISEASE CLASS',
             fontsize=15, fontweight='bold', color=PALETTE['text'])

# Age -----------------------------------------------------------------------------------------
ax = axes[0]
data_age = [df_raw[df_raw['Liver_Disease_Class'] == c]['Age'].dropna()
            for c in CLASS_COLORS]
bp = ax.boxplot(data_age, patch_artist=True, notch=True, vert=True,
                medianprops={'color': PALETTE['text'], 'linewidth': 2},
                whiskerprops={'color': PALETTE['subtext']},
                capprops={'color': PALETTE['subtext']},
                flierprops={'marker': 'o', 'markersize': 3,
                            'markerfacecolor': PALETTE['subtext'], 'alpha': 0.4})
for patch, clr in zip(bp['boxes'], CLASS_COLORS.values()):
    patch.set_facecolor(clr)
    patch.set_alpha(0.7)

ax.set_facecolor(PALETTE['surface'])
ax.set_xticks(range(1, len(CLASS_COLORS)+1))
ax.set_xticklabels([c.replace(' ', '\n') for c in CLASS_COLORS],
                   fontsize=7, color=PALETTE['subtext'])
ax.set_ylabel('Age', color=PALETTE['text'])
ax.set_title('Age by Disease Class', color=PALETTE['text'], fontsize=12)
ax.spines[:].set_color(PALETTE['border'])
ax.tick_params(colors=PALETTE['subtext'])
ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])

# Gender -----------------------------------------------------------------------------------
ax2 = axes[1]
g_ct = pd.crosstab(df_raw['Gender'], df_raw['Liver_Disease_Class'])
bottom = np.zeros(len(g_ct.columns))
for gender, clr in zip(g_ct.index, [PALETTE['accent1'], PALETTE['accent5']]):
    vals = g_ct.loc[gender].values
    ax2.bar(g_ct.columns, vals, bottom=bottom, color=clr, alpha=0.8,
            label=gender, edgecolor=PALETTE['bg'], linewidth=0.8)
    bottom += vals

ax2.set_facecolor(PALETTE['surface'])
ax2.set_xticklabels(g_ct.columns, rotation=30, ha='right', fontsize=8,
                    color=PALETTE['subtext'])
ax2.set_ylabel('Count', color=PALETTE['text'])
ax2.set_title('Gender by Disease Class', color=PALETTE['text'], fontsize=12)
ax2.legend(facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])
ax2.spines[:].set_color(PALETTE['border'])
ax2.tick_params(colors=PALETTE['subtext'])
ax2.grid(axis='y', alpha=0.3, color=PALETTE['border'])

plt.tight_layout()
save('06_age_gender')

print('  All EDA plots saved ✓')


# 4. taining ml --------------------------------------------------------------------------------------

section('4 / TRAINING ML MODELS')

models = {
    'Logistic Regression':  LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':        DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest':        RandomForestClassifier(n_estimators=150, max_depth=15,
                                                    random_state=42, n_jobs=-1),
    'SVM':                  SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'K-Nearest Neighbors':  KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
}

results = {}
for name, model in models.items():
    print(f'  Training {name}...', end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cv   = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy', n_jobs=-1).mean()

    results[name] = {'Accuracy': acc, 'Precision': prec,
                     'Recall': rec, 'F1 Score': f1,
                     'CV Score': cv, 'y_pred': y_pred}
    print(f'Acc={acc:.4f}  F1={f1:.4f}  CV={cv:.4f}')


# 5. EVALUATION gr------------------------------------------------------------------------------------

section('5 / MODEL EVALUATION')

# Plot 7: comparison Bar Chart --------------------------------------------------------------------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'CV Score']
model_names = list(results.keys())
short_names = ['LR', 'DT', 'RF', 'SVM', 'KNN']

fig, axes = plt.subplots(1, len(metrics), figsize=(20, 6))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('MODEL PERFORMANCE COMPARISON',
             fontsize=16, fontweight='bold', color=PALETTE['text'], y=1.02)

bar_colors = [PALETTE['accent1'], PALETTE['accent4'], PALETTE['accent2'],
              PALETTE['accent3'], PALETTE['accent5']]

for ax, metric in zip(axes, metrics):
    vals = [results[m][metric] for m in model_names]
    best_idx = np.argmax(vals)
    clrs = [PALETTE['accent2'] if i == best_idx else bar_colors[i]
            for i in range(len(vals))]
    bars = ax.bar(short_names, vals, color=clrs,
                  edgecolor=PALETTE['bg'], linewidth=1.2, width=0.65)
    ax.set_facecolor(PALETTE['surface'])
    ax.set_title(metric, color=PALETTE['text'], fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel('Score', color=PALETTE['text'], fontsize=9)
    ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
    ax.spines[:].set_color(PALETTE['border'])
    ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                color=PALETTE['text'], fontsize=8, fontweight='bold')

plt.tight_layout()
save('07_model_comparison')

#  Plot 8: Leaderboard---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(PALETTE['bg'])
ax.set_facecolor(PALETTE['surface'])

x = np.arange(len(model_names))
width = 0.14
metric_colors = [PALETTE['accent1'], PALETTE['accent4'],
                 PALETTE['accent2'], PALETTE['accent3'], PALETTE['accent5']]

for i, (metric, clr) in enumerate(zip(metrics, metric_colors)):
    vals = [results[m][metric] for m in model_names]
    ax.bar(x + i * width - width * 2, vals, width,
           color=clr, label=metric, edgecolor=PALETTE['bg'], linewidth=0.7)

ax.set_xticks(x)
ax.set_xticklabels(short_names, color=PALETTE['subtext'], fontsize=11)
ax.set_ylabel('Score', color=PALETTE['text'], fontsize=11)
ax.set_ylim(0.45, 1.08)
ax.set_title('ALL METRICS SIDE-BY-SIDE',
             color=PALETTE['text'], fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=9,
          facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])
ax.spines[:].set_color(PALETTE['border'])
ax.tick_params(colors=PALETTE['subtext'])
ax.grid(axis='y', alpha=0.3, color=PALETTE['border'])

plt.tight_layout()
save('08_all_metrics')

# Plot 9: confu matrix-------------------------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(26, 5))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle('CONFUSION MATRICES — ALL MODELS',
             fontsize=16, fontweight='bold', color=PALETTE['text'])

short_class = ['Healthy', 'NAFLD', 'Alcoholic', 'General', 'Cirrhosis']
cmap_cm = LinearSegmentedColormap.from_list('cm', [PALETTE['surface'], PALETTE['accent1']])

for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, ax=ax, cmap=cmap_cm, annot=True, fmt='.2f',
                annot_kws={'size': 8, 'color': PALETTE['text']},
                linewidths=0.5, linecolor=PALETTE['border'],
                xticklabels=short_class, yticklabels=short_class,
                cbar=False)
    ax.set_facecolor(PALETTE['surface'])
    short = {'Logistic Regression':'LR','Decision Tree':'DT',
             'Random Forest':'RF','SVM':'SVM','K-Nearest Neighbors':'KNN'}
    ax.set_title(f"{short[name]}\nAcc={res['Accuracy']:.3f}",
                 color=PALETTE['text'], fontsize=10, fontweight='bold')
    ax.tick_params(colors=PALETTE['subtext'], labelsize=7)
    ax.set_xlabel('Predicted', color=PALETTE['subtext'], fontsize=8)
    ax.set_ylabel('Actual', color=PALETTE['subtext'], fontsize=8)

plt.tight_layout()
save('09_confusion_matrices')

# Plot 10: random fo ----------------------------------------------------------------------------
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values('Importance', ascending=True).tail(20)

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(PALETTE['bg'])
ax.set_facecolor(PALETTE['surface'])

# Color-code ---------------------------------------------------------------------------
def feat_color(f):
    if f in ['ALT', 'AST', 'Bilirubin', 'Albumin', 'Alk_Phosphatase',
             'GGT', 'INR', 'Platelets', 'Triglycerides']:
        return PALETTE['accent3']
    elif f in ['BMI', 'Waist_Circumference', 'Obesity_Class']:
        return PALETTE['accent4']
    elif f in ['Alcohol_Consumption', 'Smoking_Status']:
        return PALETTE['accent5']
    elif f.startswith('Sym_'):
        return PALETTE['accent2']
    else:
        return PALETTE['accent1']

colors_feat = [feat_color(f) for f in feat_df['Feature']]
bars = ax.barh(feat_df['Feature'], feat_df['Importance'],
               color=colors_feat, edgecolor=PALETTE['bg'], linewidth=0.8)

ax.set_xlabel('Feature Importance Score', color=PALETTE['text'], fontsize=12)
ax.set_title('FEATURE IMPORTANCE — RANDOM FOREST\n(Top 20 Features)',
             color=PALETTE['text'], fontsize=14, fontweight='bold')
ax.spines[:].set_color(PALETTE['border'])
ax.tick_params(colors=PALETTE['subtext'], labelsize=9)
ax.grid(axis='x', alpha=0.3, color=PALETTE['border'])

# Legend item---------------------------------------------------------------------------------------------
legend_items = [
    mpatches.Patch(color=PALETTE['accent3'], label='Blood / Enzyme Tests'),
    mpatches.Patch(color=PALETTE['accent4'], label='BMI / Obesity'),
    mpatches.Patch(color=PALETTE['accent5'], label='Alcohol / Smoking'),
    mpatches.Patch(color=PALETTE['accent2'], label='Symptoms'),
    mpatches.Patch(color=PALETTE['accent1'], label='Demographics'),
]
ax.legend(handles=legend_items, loc='lower right', fontsize=9,
          facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])

plt.tight_layout()
save('10_feature_importance')


# 6. FINAL TABLE-----------------------------------------------------------------------------------

section('6 / FINAL RESULTS SUMMARY')

results_df = pd.DataFrame(
    {m: {k: v for k, v in r.items() if k != 'y_pred'} for m, r in results.items()}
).T.sort_values('F1 Score', ascending=False)

print(f'\n{"Model":<25}  {"Accuracy":>9}  {"Precision":>10}  {"Recall":>8}  '
      f'{"F1 Score":>9}  {"CV Score":>9}')
print('─' * 80)
for idx, row in results_df.iterrows():
    star = ' ★' if idx == results_df.index[0] else ''
    print(f'{idx:<25}{star}  {row["Accuracy"]:>9.4f}  {row["Precision"]:>10.4f}  '
          f'{row["Recall"]:>8.4f}  {row["F1 Score"]:>9.4f}  {row["CV Score"]:>9.4f}')

best_model_name = results_df.index[0]
best = results_df.iloc[0]
print(f'\n  ★ BEST MODEL: {best_model_name}')
print(f'    Accuracy  : {best["Accuracy"]:.4f}')
print(f'    F1 Score  : {best["F1 Score"]:.4f}')
print(f'    CV Score  : {best["CV Score"]:.4f}')

# Plot 11: Dashboard  ----------------------------------------------------------------------
fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor(PALETTE['bg'])

# Title -------------------------------------------------------------------------------------------
fig.text(0.5, 0.97, 'LIVER DISEASE PREDICTION — PROJECT SUMMARY DASHBOARD',
         ha='center', va='top', fontsize=18, fontweight='bold',
         color=PALETTE['text'])

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                       top=0.91, bottom=0.08)

# Subplot A: Class counts ─-------------------------------------------------------------------------
ax_a = fig.add_subplot(gs[0, 0])
ax_a.set_facecolor(PALETTE['surface'])
cls_vals = df_raw['Liver_Disease_Class'].value_counts()
clrs_a   = [CLASS_COLORS[c] for c in cls_vals.index]
ax_a.barh(cls_vals.index, cls_vals.values, color=clrs_a,
          edgecolor=PALETTE['bg'], linewidth=1.2)
ax_a.set_title('Dataset Class Balance', color=PALETTE['text'],
               fontsize=11, fontweight='bold')
ax_a.set_xlabel('Patients', color=PALETTE['subtext'], fontsize=9)
ax_a.tick_params(colors=PALETTE['subtext'], labelsize=8)
ax_a.spines[:].set_color(PALETTE['border'])
ax_a.grid(axis='x', alpha=0.3, color=PALETTE['border'])

# ─ Subplot B: enzyme differences -------------------------------------------------------------------
ax_b = fig.add_subplot(gs[0, 1])
ax_b.set_facecolor(PALETTE['surface'])
for col, clr in zip(['ALT', 'AST', 'GGT'], [PALETTE['accent1'], PALETTE['accent3'], PALETTE['accent4']]):
    means = [df_raw[df_raw['Liver_Disease_Class'] == c][col].mean() for c in CLASS_COLORS]
    ax_b.plot(list(CLASS_COLORS.keys()), means, 'o-', color=clr, label=col,
              linewidth=2, markersize=7)
ax_b.set_facecolor(PALETTE['surface'])
ax_b.set_xticklabels([c.replace(' ', '\n') for c in CLASS_COLORS],
                     fontsize=7, color=PALETTE['subtext'])
ax_b.set_xticks(range(len(CLASS_COLORS)))
ax_b.set_ylabel('Mean Level', color=PALETTE['text'], fontsize=9)
ax_b.set_title('Mean Enzyme Levels', color=PALETTE['text'],
               fontsize=11, fontweight='bold')
ax_b.legend(fontsize=8, facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])
ax_b.spines[:].set_color(PALETTE['border'])
ax_b.tick_params(colors=PALETTE['subtext'])
ax_b.grid(alpha=0.3, color=PALETTE['border'])

# ─ Subplot C: ac radar bars ------------------------------------------------------------
ax_c = fig.add_subplot(gs[0, 2])
ax_c.set_facecolor(PALETTE['surface'])
accs = [results[m]['Accuracy'] for m in model_names]
f1s  = [results[m]['F1 Score'] for m in model_names]
x_c  = np.arange(len(short_names))
ax_c.bar(x_c - 0.18, accs, 0.35, color=PALETTE['accent1'], label='Accuracy',
         edgecolor=PALETTE['bg'])
ax_c.bar(x_c + 0.18, f1s,  0.35, color=PALETTE['accent2'], label='F1 Score',
         edgecolor=PALETTE['bg'])
ax_c.set_xticks(x_c); ax_c.set_xticklabels(short_names, color=PALETTE['subtext'])
ax_c.set_ylim(0.5, 1.05)
ax_c.set_title('Accuracy vs F1 Score', color=PALETTE['text'],
               fontsize=11, fontweight='bold')
ax_c.legend(fontsize=8, facecolor=PALETTE['surface2'], labelcolor=PALETTE['text'])
ax_c.spines[:].set_color(PALETTE['border'])
ax_c.tick_params(colors=PALETTE['subtext'])
ax_c.grid(axis='y', alpha=0.3, color=PALETTE['border'])

# ─ Subplot D: confusion -----------------------------------------------------------------
ax_d = fig.add_subplot(gs[1, 0])
cm_best = confusion_matrix(y_test, results[best_model_name]['y_pred'])
cm_best_n = cm_best.astype(float) / cm_best.sum(axis=1, keepdims=True)
sns.heatmap(cm_best_n, ax=ax_d, cmap=cmap_cm, annot=True, fmt='.2f',
            annot_kws={'size': 8, 'color': PALETTE['text']},
            linewidths=0.5, linecolor=PALETTE['border'],
            xticklabels=short_class, yticklabels=short_class, cbar=False)
ax_d.set_facecolor(PALETTE['surface'])
ax_d.set_title(f'Best Model Confusion\n({best_model_name})',
               color=PALETTE['text'], fontsize=11, fontweight='bold')
ax_d.tick_params(colors=PALETTE['subtext'], labelsize=8)
ax_d.set_xlabel('Predicted', color=PALETTE['subtext'], fontsize=9)
ax_d.set_ylabel('Actual', color=PALETTE['subtext'], fontsize=9)

# ─ Subplot  -----------------------------------------------------------------------------
ax_e = fig.add_subplot(gs[1, 1])
ax_e.set_facecolor(PALETTE['surface'])
top10 = feat_df.tail(10)
clrs_e = [feat_color(f) for f in top10['Feature']]
ax_e.barh(top10['Feature'], top10['Importance'], color=clrs_e,
          edgecolor=PALETTE['bg'], linewidth=0.8)
ax_e.set_title('Top 10 Features (RF)', color=PALETTE['text'],
               fontsize=11, fontweight='bold')
ax_e.set_xlabel('Importance', color=PALETTE['subtext'], fontsize=9)
ax_e.tick_params(colors=PALETTE['subtext'], labelsize=8)
ax_e.spines[:].set_color(PALETTE['border'])
ax_e.grid(axis='x', alpha=0.3, color=PALETTE['border'])

# ─ Subplot  -----------------------------------------------------------------------------
ax_f = fig.add_subplot(gs[1, 2])
ax_f.set_facecolor(PALETTE['surface'])
ax_f.axis('off')

kpis = [
    ('BEST MODEL',  best_model_name,                        PALETTE['accent1']),
    ('ACCURACY',    f'{best["Accuracy"]*100:.2f}%',         PALETTE['accent2']),
    ('F1 SCORE',    f'{best["F1 Score"]*100:.2f}%',         PALETTE['accent3']),
    ('CV SCORE',    f'{best["CV Score"]*100:.2f}%',         PALETTE['accent4']),
    ('DATASET',     '5,500 patients · 33 features',         PALETTE['accent5']),
    ('CLASSES',     '5 disease categories',                 PALETTE['subtext']),
]
for i, (label, value, clr) in enumerate(kpis):
    y_pos = 0.92 - i * 0.155
    ax_f.text(0.05, y_pos, label, transform=ax_f.transAxes,
              fontsize=8, color=PALETTE['subtext'], fontweight='bold')
    ax_f.text(0.05, y_pos - 0.07, value, transform=ax_f.transAxes,
              fontsize=11, color=clr, fontweight='bold')
    ax_f.plot([0.03, 0.97], [y_pos - 0.095, y_pos - 0.095],
              color=PALETTE['border'], linewidth=0.5,
              transform=ax_f.transAxes)

ax_f.set_title('Project KPIs', color=PALETTE['text'],
               fontsize=11, fontweight='bold')

save('11_dashboard')


# 7. CLASSIFICATION REPORT ---------------------------------------------------------------

section(f'7 / CLASSIFICATION REPORT — {best_model_name}')
print(classification_report(y_test, results[best_model_name]['y_pred'],
                            target_names=class_names))

section('DONE — ALL OUTPUTS SAVED TO plots/')
print()

for f in sorted(os.listdir('plots')):
    print(f'  📊 {f}')

# 8. end prediction --------------------------------------------------------------

section("8 / REAL-TIME PREDICTION MODE")



# mappings (used in training)
gender_map = {"male":0, "female":1}

diet_map = {"poor":0, "average":1, "healthy":2}

activity_map = {
    "sedentary":0,
    "low":1,
    "moderate":2,
    "high":3
}

obesity_map = {
    "underweight":0,
    "normal":1,
    "overweight":2,
    "obesity i":3,
    "obesity ii":4,
    "obesity iii":5
}

alcohol_map = {
    "none":0,
    "low":1,
    "moderate":2,
    "high":3
}

smoking_map = {
    "never":0,
    "former":1,
    "current":2
}

while True:
    try:

        user_input = []

        for feature in X.columns:

            if feature == "Gender":
                val = input("Gender (Male/Female): ").lower()
                user_input.append(gender_map[val])

            elif feature == "Diet_Quality":
                val = input("Diet Quality (Poor/Average/Healthy): ").lower()
                user_input.append(diet_map[val])

            elif feature == "Physical_Activity":
                val = input("Physical Activity (Sedentary/Low/Moderate/High): ").lower()
                user_input.append(activity_map[val])

            elif feature == "Obesity_Class":
                val = input("Obesity Class (Underweight/Normal/Overweight/Obesity I/Obesity II/Obesity III): ").lower()
                user_input.append(obesity_map[val])

            elif feature == "Alcohol_Consumption":
                val = input("Alcohol Consumption (None/Low/Moderate/High): ").lower()
                user_input.append(alcohol_map[val])

            elif feature == "Smoking_Status":
                val = input("Smoking Status (Never/Former/Current): ").lower()
                user_input.append(smoking_map[val])

            else:
                val = float(input(f"{feature}: "))
                user_input.append(val)

        user_array = np.array(user_input).reshape(1, -1)

        user_scaled = scaler.transform(user_array)

        prediction = models[best_model_name].predict(user_scaled)

        predicted_class = target_le.inverse_transform(prediction)

        print("\n-------------------------------------")
        print("PREDICTION RESULT")
        print("-------------------------------------")
        print(f"Predicted Liver Condition: {predicted_class[0]}")
        print("-------------------------------------\n")

        again = input("Test another patient? (y/n): ").lower()

        if again != "y":
            break

    except Exception as e:
        print("Invalid input. Please try again.")
        print(e)