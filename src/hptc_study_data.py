
import pickle
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops

MICCAI_LABELS_IDS = {
    "Background": 0,
    "BrainStem": 1,
    "Chiasm": 2,
    "Mandible": 3,
    "Opt_Nrv_L": 4,
    "Opt_Nrv_R": 5,
    "Parotid_L": 6,
    "Parotid_R": 7,
    "Submandibular_L": 8,
    "Submandibular_R": 9,
}

HPTC_TO_MICCAI_LABELS = {
    "Background": "Background",
    "Brainstem": "BrainStem",
    "OpticChiasm": "Chiasm",
    "Bone_Mandible": "Mandible",
    "OpticNrv_L": "Opt_Nrv_L",
    "OpticNrv_R": "Opt_Nrv_R",
    "Parotid_L": "Parotid_L",
    "Parotid_R": "Parotid_R",
    "Glnd_Submand_L": "Submandibular_L",
    "Glnd_Submand_R": "Submandibular_R",
}

struct_id_to_name = {
    0: "Background",
    1: "BrainStem",
    3: "Mandible",
    6: "Parotid_L",
    7: "Parotid_R",
    8: "Submand_L",
    9: "Submand_R"

}

def load_study_data(patient_id="HCAI-001_CT0",
                    data_dir="/Users/chadepl/data/HCAI/HPTC/hcai-data-dl",
                    pickle_dir="/Users/chadepl/git/study-prioritized-error-finding/data"):

    data_dir = Path(data_dir).joinpath(patient_id)
    pickle_path = Path(pickle_dir).joinpath(f"{patient_id}.pkl")

    if pickle_path.exists():
        print(f"Pickled file for study {patient_id} exists, loading it ...")
        with open(pickle_path, "rb") as f:
            to_pickle = pickle.load(f)
        print(f"Successfully loaded study from {pickle_path}")
    else:
        print(f"Pickled file for study {patient_id} does not exist, processing it ...")

        if not data_dir.exists():
            raise Exception(f"Does not exist: {data_dir}")

        # - img (CT scan)
        img = sitk.ReadImage(data_dir.joinpath("img.nrrd"), outputPixelType=sitk.sitkInt64)
        img = sitk.GetArrayFromImage(img)

        # - GT segmentation
        seg_gt = sitk.ReadImage(data_dir.joinpath("APPROVED-mask.nrrd"), outputPixelType=sitk.sitkInt8)
        seg_gt = sitk.GetArrayFromImage(seg_gt)

        # - dose
        dose = sitk.ReadImage(data_dir.joinpath("dose.nrrd"))
        dose = sitk.GetArrayFromImage(dose)

        # - predicted segmentations
        seg_dir = data_dir.joinpath(
            "Grid_FocusNetFlipV21632GNorm_FixedKL010_33samB214014040PreGridNorm_WithNegCEScalar10_seed42")
        path_samples_oh = seg_dir.joinpath("samples_onehot")
        path_samples_lm = seg_dir.joinpath("samples_labelmap")

        ###
        ###

        mean = np.zeros_like(np.load(str(path_samples_oh.joinpath("seg-pred-0.npz")))["arr_0"])
        std = np.zeros_like(mean)
        count = 0
        for s in path_samples_oh.glob("seg-pred-*"):  # loop for mean computation
            mean += np.load(str(s))["arr_0"]
            count += 1
        mean = mean / count
        for s in path_samples_oh.glob("seg-pred-*"):  # loop for std computation
            std += np.square(np.load(str(s))["arr_0"] - mean)
        std = np.sqrt(std / count)
        mean = mean.transpose()
        std = std.transpose()

        # Finding the final seg
        # - Find median of labelmap samples
        median = []
        for p in path_samples_lm.glob("seg-pred*"):
            lm = sitk.ReadImage(str(p))
            median.append(np.expand_dims(sitk.GetArrayFromImage(lm), axis=0))
        median = np.stack(median, axis=0)
        median = np.median(median, axis=0).squeeze()
        median_oh = [(median == i).astype(np.uint8) for i in range(len(MICCAI_LABELS_IDS))]

        # - Connected components analysis
        connected_oh = []
        ccif = sitk.ConnectedComponentImageFilter()
        lssif = sitk.LabelShapeStatisticsImageFilter()
        for moh in median_oh:
            components_vol = ccif.Execute(sitk.GetImageFromArray(moh))
            lssif.Execute(components_vol)
            largest_label = -1
            largest_num_pixels = 0
            for l in lssif.GetLabels():
                num_pixels = lssif.GetNumberOfPixels(l)
                if num_pixels > largest_num_pixels:
                    largest_label = l
                    largest_num_pixels = num_pixels
            components_vol = sitk.GetArrayFromImage(components_vol)
            components_vol = (components_vol == largest_label).astype(np.uint8)
            connected_oh.append(components_vol)

        # - Get final lm
        final_lm = np.zeros_like(connected_oh[0])
        for i in range(1, len(connected_oh)):
            coh = connected_oh[i]
            final_lm[coh == 1] = coh[coh == 1] * i

        proc_lm = final_lm.astype(np.uint8)

        ###
        ###

        # seg_stack = []
        # for i in range(10):  # we have 10 candidates
        #     segs = np.load(seg_dir.joinpath(f"seg-pred-{i}.npz"))["arr_0"]
        #     segs = np.moveaxis(segs, [3, 2, 1, 0], [0, 1, 2, 3])
        #     segs = np.expand_dims(segs, axis=0)
        #     seg_stack.append(segs)
        # seg_stack = np.concatenate(seg_stack, axis=0)
        #
        # mean = seg_stack.mean(axis=0)
        # std = seg_stack.std(axis=0)
        # # - - Get final segmentation
        # lm = np.argmax(seg_stack, axis=1)
        # lm = np.median(lm, axis=0).astype(int)
        #
        # proc_lm = np.zeros(img.shape)
        # for i in range(1, 10):
        #     lab = label((lm == i).astype(int))
        #     lab = (lab == np.argmax(np.bincount(lab.flat)[1:]) + 1).astype(int)
        #     proc_lm += i * lab
        # proc_lm = proc_lm.astype(int)
        #
        # # remove structures for which there is no GT
        # for i in np.unique(proc_lm):
        #     if i not in list(np.unique(seg_gt)):
        #         proc_lm[proc_lm == i] = 0  # set to background

        # ANALYSIS BASED ON BBOXES
        bboxes = dict()
        for i in np.unique(proc_lm):
            if i != 0:
                rps = regionprops((proc_lm == i).astype(int))
                bbox = list(rps[0]["bbox"])
                bboxes[i] = dict(a0_0=bbox[0], a0_1=bbox[3], a1_0=bbox[1], a1_1=bbox[4], a2_0=bbox[2], a2_1=bbox[5])

        struct_name_to_id = dict()
        for k, v in struct_id_to_name.items():
            struct_name_to_id[v] = k

        to_pickle = dict(img=img,
                         seg_gt=seg_gt,
                         dose=dose,
                         mean=mean,
                         std=std,
                         lm=proc_lm,
                         bboxes=bboxes,
                         struct_id_to_name=struct_id_to_name,
                         struct_name_to_id=struct_name_to_id)

        with open(pickle_path, "wb") as f:
            pickle.dump(to_pickle, f)
        print(f"Successfully persisted study at {pickle_path}")

    return to_pickle


def load_study_candidates(patient_id="HCAI-001_CT0", oar="BrainStem",
                    data_dir="/Users/chadepl/data/HCAI/HPTC/hcai-data-dl",
                    pickle_dir="/Users/chadepl/git/study-prioritized-error-finding/data"):
    data_dir = Path(data_dir).joinpath(patient_id)
    oar_id = MICCAI_LABELS_IDS[oar]
    
    # - predicted segmentations
    seg_dir = data_dir.joinpath(
        "Grid_FocusNetFlipV21632GNorm_FixedKL010_33samB214014040PreGridNorm_WithNegCEScalar10_seed42")
    path_samples_oh = seg_dir.joinpath("samples_onehot")
    path_samples_lm = seg_dir.joinpath("samples_labelmap")    
    
    out = []
    for p in path_samples_lm.glob("seg-pred*"):
        lm = sitk.ReadImage(str(p))
        arr = np.expand_dims(sitk.GetArrayFromImage(lm), axis=0)
        out.append((arr == oar_id).astype(int))

    return np.array(out).squeeze()