from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import functools

# DAHANCA based guidelines
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwijjoye1oj_AhW89bsIHTb3AiIQFnoECAgQAQ&url=https%3A%2F%2Fwww.dahanca.dk%2Fuploads%2FTilFagfolk%2FGuideline%2FGUID_DAHANCA_Radiotherapy_guidelines_2020.pdf&usg=AOvVaw3_7NOIbrUZ9LlCkJ-DmRXF
struct_gl = dict(
    BrainStem=dict(metric="max", value=54),
    Mandible=dict(metric="max", value=72),
    Parotid_L=dict(metric="mean", value=26),  # 20 Gy if contralateral
    Parotid_R=dict(metric="mean", value=26),
    Submandibular_L=dict(metric="mean", value=35),
    Submandibular_R=dict(metric="mean", value=35)
)

BASE_DIR = Path(f"results_data/simulation-study")


def make_tex_overview_table(df, return_df=False, file_buf=None):

    #############################################
    # Table of structures (total time analysis) #
    #############################################

    # This table shows the global picture, therefore only total time
    # In the plots below, we perform a more fine grained analysis of ANA vs EDIT
    # OAR | Dose limit | # Slices | T_baseline | T_error | T_dose | T_combined

    # might need to join 3 df
    # df1: general info about structures
    # df2: total time per structure per condition
    # summary per condition across patients
    # the index we use to concatenate dfs is the structure name

    row_mult = df["should_review"].astype(float)  # 0 if false and 1 if true
    df["t_analysis"] = df["t_analysis"] * row_mult
    df["t_editing"] = df["t_editing"] * row_mult
    df["t_total"] = df["t_total"] * row_mult
    df["addressed_error"] = df["slices_sum_error"] * row_mult # we assume that all error was addressed in slices that were reviewed

    df_baseline = df.loc[df["workflow"] == "baseline", :]
    df_error_easy = df.loc[np.logical_and(df["workflow"] == "error", df["scenario"] == "easy"), :]    
    df_error_hard = df.loc[np.logical_and(df["workflow"] == "error", df["scenario"] == "hard"), :]
    df_dose_easy = df.loc[np.logical_and(df["workflow"] == "dose", df["scenario"] == "easy"), :]
    df_dose_hard = df.loc[np.logical_and(df["workflow"] == "dose", df["scenario"] == "hard"), :]
    df_table = df.copy()#pd.concat([df_baseline, df_error, df_dose], axis=0)

    # First column group: dose limit
    df1 = pd.DataFrame(struct_gl).T
    df1.index.name = "oar"
    df1 = df1.drop(["metric",], axis=1)
    df1.columns = ["dose_limit",]

    # Second column group: number of slices
    def get_df_slice_num(df_slice, prefix):
        df_slice = df_slice.loc[df_slice["sim_run_id"] == 0, :]
        df_slice = df_slice.groupby(["oar", "patient"]).sum().loc[:, "should_review"].reset_index()
        df_slice = df_slice.rename(dict(should_review="slice_num"), axis=1)
        df_slice_mean = df_slice.groupby(["oar"])["slice_num"].mean()
        df_slice_std = df_slice.groupby(["oar"])["slice_num"].std()
        df_slice = pd.concat([df_slice_mean, df_slice_std], axis=1, join="inner")
        df_slice.columns = ["mean_num_slices", "std_num_slices"]
        df_slice[f"num_slices_{prefix}"] = df_slice.apply(lambda r: f"{r.mean_num_slices:.0f} $\pm$ {r.std_num_slices:.0f}", axis=1)
        df_slice = df_slice.drop(["mean_num_slices", "std_num_slices"], axis=1)
        return df_slice

    df_slice_baseline = get_df_slice_num(df_baseline, "baseline")
    df_slice_error = get_df_slice_num(df_error_easy, "error")
    df_slice_dose = get_df_slice_num(df_dose_easy, "dose")

    num_slices_baseline_patients = df_baseline.loc[df_baseline["sim_run_id"] == 0, :].groupby(["patient"]).sum().loc[:, "should_review"].reset_index()["should_review"]
    num_slices_error_patients = df_error_easy.loc[df_error_easy["sim_run_id"] == 0, :].groupby(["patient"]).sum().loc[:, "should_review"].reset_index()["should_review"]
    num_slices_dose_patients = df_dose_easy.loc[df_dose_easy["sim_run_id"] == 0, :].groupby(["patient"]).sum().loc[:, "should_review"].reset_index()["should_review"]

    df1 = pd.concat([df1, df_slice_baseline], axis=1, join="inner")
    df1 = pd.concat([df1, df_slice_error], axis=1, join="inner")
    df1 = pd.concat([df1, df_slice_dose], axis=1, join="inner")

    # Third column group: timings
    t_key = "t_total"
    df_times_per_run = df_table.groupby(["fn", "patient", "oar", "sim_run_id"])[t_key].sum()
    df_times_per_run = df_times_per_run.reset_index()
    df_times_per_run = df_times_per_run.rename({t_key:"t_per_run"}, axis=1)
    df_time_mean = df_times_per_run.groupby(["fn", "oar"])["t_per_run"].mean()
    df_time_std = df_times_per_run.groupby(["fn", "oar"])["t_per_run"].std()
    df3 = pd.concat([df_time_mean, df_time_std], axis=1, join="inner")
    df3.columns = [f"mean_{t_key}", f"std_{t_key}"]
    df3[t_key] = df3.apply(lambda r: f"{r[f'mean_{t_key}']:.0f} $\pm$ {r[f'std_{t_key}']:.0f}", axis=1)
    df3 = df3.drop([f"mean_{t_key}", f"std_{t_key}"], axis=1).reset_index()
    df3 = df3.pivot(columns=["fn"], index=["oar"])
    df3.columns = df3.columns.get_level_values(1)
    df3 = df3.loc[:, ["baseline-None", "error-easy", "error-hard", "dose-easy", "dose-hard"]]

    df1 = pd.concat([df1, df3], axis=1, join="inner")

    # Fourth column group: qualities/errors
    
    err_key = "perc_addressed_error_per_run"
    df_addressed_error_per_run = df_table.groupby(["workflow", "patient", "oar", "sim_run_id"])[["slices_sum_error", "addressed_error"]].sum()
    df_addressed_error_per_run = df_addressed_error_per_run.reset_index()
    df_addressed_error_per_run[err_key] = (df_addressed_error_per_run["addressed_error"]/df_addressed_error_per_run["slices_sum_error"]) * 100    
    df_addressed_error_per_run_mean = df_addressed_error_per_run.groupby(["workflow", "oar"])[err_key].mean()
    df_addressed_error_per_run_std = df_addressed_error_per_run.groupby(["workflow", "oar"])[err_key].std()
    df4 = pd.concat([df_addressed_error_per_run_mean, df_addressed_error_per_run_std], axis=1, join="inner")
    df4.columns = [f"mean_{err_key}", f"std_{err_key}"]
    df4[err_key] = df4.apply(lambda r: f"{r[f'mean_{err_key}']:.0f} $\pm$ {r[f'std_{err_key}']:.0f}", axis=1)
    df4 = df4.drop([f"mean_{err_key}", f"std_{err_key}"], axis=1).reset_index()
    df4 = df4.pivot(columns=["workflow"], index=["oar"])
    df4.columns = df4.columns.get_level_values(1)

    df4 = df4.loc[:, ["baseline", "error", "dose"]]

    df1 = pd.concat([df1, df4], axis=1, join="inner")

    # - df4: last row with per column summary
    df_times_per_patient = df_times_per_run.groupby(["fn", "patient", "sim_run_id"])["t_per_run"].sum().reset_index()
    mean_df_ptime = df_times_per_patient.groupby(["fn"])["t_per_run"].mean()
    std_df_ptime = df_times_per_patient.groupby(["fn"])["t_per_run"].std()

    df_addressed_error_per_patient = df_addressed_error_per_run.groupby(["workflow", "patient", "sim_run_id"])[err_key].mean().reset_index()
    mean_df_perr = df_addressed_error_per_patient.groupby(["workflow"])[err_key].mean()
    std_df_perr = df_addressed_error_per_patient.groupby(["workflow"])[err_key].std()

    mean_df_ptime = pd.DataFrame(mean_df_ptime)    
    std_df_ptime = pd.DataFrame(std_df_ptime)
    mean_df_perr = pd.DataFrame(mean_df_perr)    
    std_df_perr = pd.DataFrame(std_df_perr)

    df5_t = pd.concat([mean_df_ptime, std_df_ptime], axis=1, join="inner")
    df5_t.columns = [f"mean_{t_key}", f"std_{t_key}"]
    df5_t[t_key] = df5_t.apply(lambda r: f"{r[f'mean_{t_key}']:.0f} $\pm$ {r[f'std_{t_key}']:.0f}", axis=1)
    df5_t = df5_t.drop([f"mean_{t_key}", f"std_{t_key}"], axis=1).reset_index()
    df5_t["oar"] = "Total"
    df5_t = df5_t.pivot(columns=["fn"], index=["oar"])
    df5_t.columns = df5_t.columns.get_level_values(1)

    df5_perr = pd.concat([mean_df_perr, std_df_perr], axis=1, join="inner")
    df5_perr.columns = [f"mean_perr", f"std_perr"]
    df5_perr["perr"] = df5_perr.apply(lambda r: f"{r[f'mean_perr']:.0f} $\pm$ {r[f'std_perr']:.0f}", axis=1)
    df5_perr = df5_perr.drop(["mean_perr", "std_perr"], axis=1).reset_index()
    df5_perr["oar"] = "Total"
    df5_perr = df5_perr.pivot(columns=["workflow"], index=["oar"])
    df5_perr.columns = df5_perr.columns.get_level_values(1)

    df5 = df5_perr.copy()    
    df5 = pd.concat([df5, df5_t], axis=1)
    df5["num_slices_baseline"] = f"{num_slices_baseline_patients.mean():.0f} $\pm$ {num_slices_baseline_patients.std():.0f}"
    df5["num_slices_error"] = f"{num_slices_error_patients.mean():.0f} $\pm$ {num_slices_error_patients.std():.0f}"
    df5["num_slices_dose"] = f"{num_slices_dose_patients.mean():.0f} $\pm$ {num_slices_dose_patients.std():.0f}"
    df5["dose_limit"] = "-"

    df5 = df5.loc[:, df1.columns]

    df1 = pd.concat([df1, df5], axis=0)

    # - final formatting and latex export

    cols_map = {
        'dose_limit': dict(parent="REMOVE", name='$l_{\text{OAR}}$'),
        'num_slices_baseline': dict(parent="REMOVE", name='\\# Slices Baseline'),
        'num_slices_error': dict(parent="REMOVE", name='\\# Slices Error'),
        'num_slices_dose': dict(parent="REMOVE", name='\\# Slices Dose'),        
        'baseline': dict(parent="Attended Error (\%)", name='$baseline$'),
        'error': dict(parent="Attended Error (\%)", name='$error$'),
        'dose': dict(parent="Attended Error (\%)", name='$dose$'),
        'baseline-None': dict(parent="T (Seconds)", name='$baseline$'),
        'error-easy': dict(parent="T (Seconds)", name='$error(\epsilon=0)$'),
        'error-hard': dict(parent="T (Seconds)", name='$error(\epsilon=4)$'),
        'dose-easy': dict(parent="T (Seconds)", name='$dose(\epsilon=0)$'),
        'dose-hard': dict(parent="T (Seconds)", name='$dose(\epsilon=4)$'),        
    }
    df1 = df1.loc[:, list(cols_map.keys())]

    latex_df = df1.copy()
    latex_df.index.name = "OAR"
    latex_df.index = latex_df.index.map(lambda v: v.replace("_", "\\_"))
    tuples = ((cols_map[c]["parent"], cols_map[c]["name"]) for c in latex_df.columns)
    latex_df.columns = pd.MultiIndex.from_tuples(tuples)

    if return_df:
        return latex_df
    else:
        return latex_df.to_latex(buf=file_buf, escape=False)



def make_plot_ana_edit_summaries(df):

    vis_df = df.copy()
    row_mult = vis_df["should_review"].astype(float)
    vis_df["t_analysis"] = vis_df["t_analysis"] * row_mult
    vis_df["t_editing"] = vis_df["t_editing"] * row_mult
    vis_df["t_total"] = vis_df["t_total"] * row_mult
    vis_df = vis_df.groupby(["patient", "oar", "fn", "sim_run_id"]).sum(["t_analysis", "t_editing", "t_total"]).reset_index()
    vis_df = vis_df.groupby(["patient", "oar", "fn"]).mean(["t_analysis", "t_editing", "t_total"]).reset_index()

    categories = ["error-hard", "error-easy", "baseline-None", "dose-easy", "dose-hard"]
    vis_df["fn"] = pd.Categorical(vis_df["fn"], categories=categories)

    categories = list(struct_gl.keys())
    vis_df["oar"] = pd.Categorical(vis_df["oar"], categories=categories)

    vis_df = vis_df.sort_values(by = ["fn", "oar"])

    # statistical values
    select_df_1 = vis_df.groupby(["oar", "fn"]).mean(["t_analysis", "t_editing", "t_total"]).reset_index()
    # baseline_t_total = vis_df[vis_df.fn == "baseline-None"].groupby(["patient", "sim_run_id"]).sum(["t_analysis", "t_editing", "t_total"]).reset_index().loc[:, ["patient",  "t_total"]]
    # med_pat, med_t_total = baseline_t_total.sort_values("t_total").iloc[baseline_t_total.shape[0]//2].tolist()
    # select_df_1 = vis_df[vis_df["patient"] == med_pat]

    fig, axs = plt.subplots(ncols=2, layout="tight", figsize=(14,4))#, sharey=True)

    # context: all patients (we compute the mean time per patient)
    bp1 = sns.stripplot(vis_df, x="oar", y="t_analysis", hue="fn", color="lightgray", linewidth=1, jitter=False, dodge=True, alpha=0.8, marker="_", ax=axs[0])
    bp2 = sns.stripplot(vis_df, x="oar", y="t_editing", hue="fn", color="lightgray", linewidth=1, jitter=False, dodge=True, alpha=0.8, marker="_", ax=axs[1])    

    sns.stripplot(select_df_1, x="oar", y="t_analysis", hue="fn", linewidth=1, size=7, jitter=False, dodge=True, alpha=1, marker="D", ax=axs[0])
    sns.stripplot(select_df_1, x="oar", y="t_editing", hue="fn", linewidth=1, size=7, jitter=False, dodge=True, alpha=1, marker="D", ax=axs[1])    

    axs0_ylim = axs[0].get_ylim()
    axs1_ylim = axs[1].get_ylim()
    axs_ylim = [min(axs0_ylim[0], axs1_ylim[0]), max(axs0_ylim[1], axs1_ylim[1])]

    for x in range(0, len(vis_df['oar'].unique()) - 1):
        axs[0].plot([x + 0.5, x + 0.5], [0, axs_ylim[1]], linewidth=0.5, c='lightgray')
        axs[1].plot([x + 0.5, x + 0.5], [0, axs_ylim[1]], linewidth=0.5, c='lightgray')

    axs[0].get_legend().remove()
    # axs[0].get_legend().set_title("Condition")
    axs[0].set_ylabel("Total Analysis Time (Seconds)")
    axs[0].set_xlabel("")
    axs[1].get_legend().remove()
    axs[1].set_ylabel("Total Editing Time (Seconds)")
    axs[1].set_xlabel("")

    bp1.set_xticklabels(bp1.get_xticklabels(), rotation=45)
    bp2.set_xticklabels(bp2.get_xticklabels(), rotation=45)

    axs[0].set(yscale="log", ylim=[0,axs_ylim[1]])
    axs[1].set(yscale="log", ylim=[0,axs_ylim[1]])

    plt.show()
    # plt.savefig("/Users/chadepl/Downloads/simu-plot1.png", dpi=300)



if __name__ == "__main__":

    dfs = []
    for fp in BASE_DIR.rglob("*.csv"):
        print(fp)
        df = pd.read_csv(fp)
        df["fn"] = fp.stem
        dfs.append(df)
    print()

    df = pd.concat(dfs, axis=0)
    df = df.drop(df.columns[0], axis=1)
    df["t_total"] = df["t_analysis"] + df["t_editing"]

    ##################
    # Table overview #
    ##################    

    tex_overview = make_tex_overview_table(df, return_df=False, file_buf="table.txt")

    # print(tex_overview)

    make_plot_ana_edit_summaries(df)


