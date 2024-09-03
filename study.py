
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, exponnorm

import matplotlib.pyplot as plt
import seaborn as sns

from src.hptc_study_data import load_study_data, struct_id_to_name, MICCAI_LABELS_IDS


RANDOM_SEED = 42  # TODO

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

SIM_RUNS = 100

# ASSUMPTIONS
workflows = [("baseline", None), 
             ("error", "easy"), ("error", "hard"),
             ("dose", "easy"), ("dose", "hard")]

# - baseline times for analyzing slices and editing them
# - editing time is independent from the ordering
# - analysis and editing times from (Aselmaa, 2017)
ANA_T_MU = 4.2  # Baseline time (mean)
ANA_T_STD = 3.2  # Baseline time (std)
EDIT_BRUSH_SIZE = 10  # Assumed brush size in pixels
EDIT_T_MU = 0.5  # Average time that it takes to edit per region of size EDIT_BRUSH_SIZE
EDIT_T_STD = 0.1  # Std time that it takes to edit per region of size EDIT_BRUSH_SIZE

# - for the DEDS workflow, we try a range of mean slice analysis cases
# - in an ideal scenario, the user takes as much time per slice as in the baseline
# - if there is not enough user support, the time might increase
# - we model these differences via normal and expnormal dists
MU_OFFSET_EASY = 0
MU_OFFSET_HARD = 4

K_EXPNORM = 1
K_EXPNORM_EASY = 1
K_EXPNORM_HARD = 4


def run_user_sim(img, dose, error, workflow="baseline", scenario=None, err_limit=None, dose_limit=None):
    """
    User sorts slices in descending error size order.
    Then, the user navigates from the top to the bottom of the sorted list.
    Depending on the scenario, the analysis time might change.
    If early termination limit (et_limit) is not None, the workflow
    stops when a slice has a max(dose) value below the limit.
    """
    num_slices = len(img)

    slices_seq = np.arange(num_slices)
    slice_max_dose = dose.max(axis=(1, 2))[slices_seq]
    slice_sum_error = error.sum(axis=(1, 2))[slices_seq]

    t_analysis = []
    t_editing = []
        
    ana_t_mu, ana_t_std = ANA_T_MU, ANA_T_STD
    edit_t_mu, edit_t_std = EDIT_T_MU, EDIT_T_STD
    ana_t_k, edit_t_k = K_EXPNORM, K_EXPNORM
    
    # Normal distribution
    if scenario is not None:
        if scenario == "easy":
            ana_t_mu += MU_OFFSET_EASY
        if scenario == "hard":
            ana_t_mu += MU_OFFSET_HARD
    ana_t_dist = norm(ana_t_mu, ana_t_std)
    edit_t_dist = norm(edit_t_mu, edit_t_std)

    # define and setup sequence to follow 
    slices_over_limit = [True for _ in range(num_slices)] # we check all slices
    if workflow == "baseline":        
        slice_seq_sorting = slices_seq
        direction = np.random.choice([-1, 1])
        if direction == -1:
            slice_seq_sorting = slice_seq_sorting[::-1]        
    elif workflow == "error":
        if err_limit is not None:
            slices_over_limit = (slice_sum_error > err_limit).tolist()
        slice_seq_sorting = np.argsort(slice_sum_error)[::-1] # descending sum(error)-based sorting
    elif workflow == "dose":
        if dose_limit is not None:
            slices_over_limit = (slice_max_dose > dose_limit).tolist()
        slice_seq_sorting = np.argsort(slice_max_dose)[::-1] # descending max(dose)-based sorting

    slices_seq = slices_seq[slice_seq_sorting]
    flags_should_check_slices = []      

    for slice_id in slices_seq:
        should_check_slice = slices_over_limit[slice_id]

        # analyze slice
        # t = np.random.normal(ana_t_mu, ana_t_std)
        t = ana_t_dist.rvs()
        t_analysis.append(t if t >= 0 else 0)

        # edit slice
        se = slice_sum_error[slice_id]
        #t = (se / edit_brush_size) * np.random.normal(edit_t_mu, edit_t_std)
        t = edit_t_dist.rvs()
        t = (se / EDIT_BRUSH_SIZE) * t 
        t_editing.append(t if t >= 0 else 0)

        flags_should_check_slices.append(should_check_slice)

    return dict(
                slices_ids = np.arange(num_slices), # each slice visited has an id (to stablish order)
                slices_seq=slices_seq,  # followed workflow (sequence)
                t_analysis=t_analysis,  # analysis time per slice
                t_editing=t_editing,  # editing time per slice
                slices_dose=slice_max_dose[slice_seq_sorting],  # max dose per slice
                slices_error=slice_sum_error[slice_seq_sorting], # sum error per slice
                should_review=flags_should_check_slices,
                )

def process_errors_sanders(ref_labels, errors, errors_overseg, errors_underseg):
    # preprocessing of errors based on Sander's paper.
    # an error is valid if it fulfills three conditions:
    # 1. 3 voxels for under segmentation and 2 voxels for over segmentation
    #    Note: we use 2 and 2 for simplicity
    # 2. 2D connected region of size 10 of larger
    # 3. error above and below structure ends are considered
    
    from scipy.ndimage import distance_transform_edt
    from skimage import measure

    true_labels_dm = distance_transform_edt(ref_labels)
    true_labels_dm += distance_transform_edt(1 - ref_labels)

    # - Rule 1: only keep errors that are thick enough
    significant_error = np.zeros_like(errors)
    significant_error[errors_overseg == 1] = (true_labels_dm[errors_overseg == 1] > 2).astype(float)
    significant_error[errors_underseg == 1] = (true_labels_dm[errors_underseg == 1] > 2).astype(float)  # we use 2 for simplicity

    # - Rule 2: remove non-significant errors
    significant_error = significant_error.copy()
    labeling = measure.label(significant_error, background=0, connectivity=1)
    significant_error_labels = np.unique(labeling)[1:]
    for label in significant_error_labels:
        label_size = (labeling == label).sum()
        if label_size <= 10: # removes error if its not significant
            slice_ids, rows, cols = np.where(labeling == label)
            significant_error[(slice_ids, rows, cols)] = 0

    # - Rule 3: only focus on bbox where oar lies
    true_labels_slices = np.where(ref_labels.sum(axis=(1, 2)) > 0)[0]
    if true_labels_slices.size > 0:
        start_slice = true_labels_slices[0]
        end_slice = true_labels_slices[-1]
        significant_error[0:start_slice] = errors[0:start_slice]
        significant_error[end_slice:] = errors[end_slice:]

    return significant_error


if __name__ == "__main__":
    # Execute process: 1) go to next slice -> 2) analyze slice -> 3) if errors -> 4) edit -> if more slices go to (1)

    df_rows = []

    data_dir = Path("/Users/chadepl/data/HCAI/HPTC/hcai-data-dl")
    patient_ids = [p.stem for p in list(data_dir.glob("HCAI*"))]
    output_dir = Path(f"results_data/simulation-study/")

    if not output_dir.exists():
        print("[simulation_study] Results do not exist, calculating them ...")
        output_dir.mkdir(parents=True)


    # run_id | patient_name | oar_name | slices_id | slice_order | times | sum error (slice) | max dose (slice)

    for patient_id in patient_ids:        

        patient_dir = output_dir.joinpath(patient_id)
        patient_dir.mkdir(exist_ok=True)

        #####################
        # LOAD PATIENT DATA #
        #####################

        study_vols = load_study_data(patient_id=patient_id)

        img = study_vols["img"]
        dose = study_vols["dose"]
        true_labels = study_vols["seg_gt"]
        pred_labels = study_vols["lm"]
              
        for workflow in workflows:
            workflow_name, workflow_scenario = workflow
            fn = f"{workflow_name}-{workflow_scenario}"

            df_rows = []  

            print(f"Processing {patient_id} ({fn})")

            for struct_name in list(struct_gl.keys()):
                print(f" - {struct_name}")

                struct_id = MICCAI_LABELS_IDS[struct_name]

                ################
                # BBOX VOLUMES #
                ################

                bboxes = study_vols["bboxes"]
                bbox = bboxes[struct_id]

                a00, a01, a10, a11, a20, a21 = bbox["a0_0"], bbox["a0_1"], bbox["a1_0"], bbox["a1_1"], bbox["a2_0"], bbox[
                    "a2_1"]

                oar_img = img[a00:a01, a10:a11, a20:a21].astype(float)
                oar_dose = dose[a00:a01, a10:a11, a20:a21].astype(float)
                oar_pred_labels = (pred_labels == struct_id)[a00:a01, a10:a11, a20:a21].astype(float)
                oar_true_labels = (true_labels == struct_id)[a00:a01, a10:a11, a20:a21].astype(float)

                oar_error_gt = (oar_true_labels != oar_pred_labels).astype(float)
                oar_err_us = np.logical_and(oar_true_labels != oar_pred_labels, oar_true_labels == 1).astype(float)
                oar_err_os = np.logical_and(oar_true_labels != oar_pred_labels, oar_pred_labels == 1).astype(float)

                oar_limit = struct_gl[struct_name]["value"]
                
                significant_error = process_errors_sanders(oar_true_labels, oar_error_gt, oar_err_os, oar_err_us)

                # all simulation runs per workflow per patient
                for run in range(SIM_RUNS):
                    print(f" -- run {run}")
                    run_u = run_user_sim(oar_img, oar_dose, significant_error,
                                            workflow=workflow_name, scenario=workflow_scenario,
                                            err_limit=0, dose_limit=oar_limit)

                    for sid in run_u["slices_ids"]:
                        df_rows.append(dict(
                            sim_run_id=run,
                            patient=patient_id,
                            oar=struct_name,
                            workflow=workflow_name,
                            scenario=workflow_scenario,

                            num_slices=len(run_u["slices_seq"]),
                            slices_ids=run_u["slices_ids"][sid],
                            slices_seq=run_u["slices_seq"][sid],
                            t_analysis=run_u["t_analysis"][sid],
                            t_editing=run_u["t_editing"][sid],
                            slices_max_dose=run_u["slices_dose"][sid],
                            slices_sum_error=run_u["slices_error"][sid],
                            should_review=run_u["should_review"][sid],
                        ))                    

            pd.DataFrame(df_rows).to_csv(patient_dir.joinpath(f"{fn}.csv"))
            print("Done")
