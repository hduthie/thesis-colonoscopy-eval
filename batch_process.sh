#!/bin/bash

set -e  # Exit if anything fails

download_gdrive() {
    FILEID="$1"
    ZIP_PATH="$2"

    echo "⬇️ Downloading with gdown..."
    if ! gdown --id "$FILEID" -O "$ZIP_PATH"; then
        echo "❌ Failed to download: $ZIP_PATH"
        return 1
    fi

    if ! unzip -t "$ZIP_PATH" &>/dev/null; then
        echo "❌ Invalid ZIP or rate-limited: $ZIP_PATH"
        # rm -f "$ZIP_PATH"
        return 1
    fi

    echo "✅ Successfully downloaded and validated: $ZIP_PATH"
}



# List of Google Drive file IDs and video names
declare -A VIDEOS=(
    ["cecum_t1_a"]="14o6_4GQLZWx5dQq2L_drzmN_rlCT7Yhr" #need to redo - got the frame alignment wrong
    # Give it a few hours then try doign in batches - downsampled should be able to store it all
    ["cecum_t1_b"]="1z3AHdnBH_YoCMnnTfDa8SNPQYIsvaBO3"
    ["cecum_t2_a"]="13XhJIev9memFtwUf_dnjJ7o8z6O_c-xW"
    ["cecum_t2_b"]="1ykYtQGiFesev5QLfz_avYuQ5a7Zs8kgF"
    ["cecum_t2_c"]="1tNoBLpPbrQexKlnOKMK2peERn9Rj_9Dp" 
    ["cecum_t4_a"]="1FC-dR__0LVb7WH02KpUx9TZVvvvN-Gyx"
    ["cecum_t4_b"]="11SbH2AZsuciTu3iGxdXCdQZky6uDTyS5"
    ["sigmoid_t2_a"]="16epAys428g9vBQgm611TElMyAXORo7rH" 
    ["trans_t2_a"]="1ylZWWtVlXfDx9dhPIeWJ1HqDHJZ3QKdH"
    ["trans_t2_b"]="1vru228_TEgxT3aS90CmvOWsMB0RLnAxn"
    ["trans_t2_c"]="12YpowbP6zhoO_Qx9UBwhfRLXNJN1EAu4"
    ["trans_t4_a"]="18qzXMifS54jAx29yROKXXxxZg0qo-iTz"
    

    ["cecum_t3_a"]="1Uw8uCRRDm_RrgkccGbiBXZHf9P-THM2Q"
    ["desc_t4_a"]="1d9HDNg4-Og1cTWM-eIU5SWM2BMrMWhwQ" 
    ["sigmoid_t1_a"]="19VGDuZ73OWNwM8eIgDkkZYBPJPQ5BD91" 
    ["sigmoid_t3_a"]="1ZRU2KuHoc2XCbKSY_A1S-7BxEfKf9xPr" 
    ["sigmoid_t3_b"]="1XfZFAQ5_Wxle8d5wSlOCumKSg4IP8wTv" 
    ["trans_t1_a"]="1urFuVo8ZalwPmsXEZg3xzhuqhpgWV8hw"
    ["trans_t1_b"]="1hyjmd7vn86McE1nUnYCzvOm8LlyHLYwt"
    ["trans_t3_a"]="1B4aeZfAqmUJgWr8e-2YAibUe4er30ncr"
    ["trans_t4_b"]="1C-nw6MR7sxssw3LS-GpiPmwBzEYhUCHN"
    ["seq1"]="18m3Z5zJtljor_AGmPW8OgO9fRuactuNk"
    ["trans_t3_b"]="1ZpbYcDVP-sCTjjQrDc303olFgsr2nA5J"
    ["seq2"]="1kn_qevX7lLh3gWkiKAt3hgWFpy0P68s6"
    ["seq3"]="1RmOnnjJBzCMwO5gPY4h3e4MpcDDOvOb6"
    ["seq4"]="1sYps79WjJ0ETRtuWtHd_1zeuyPLtoMM7"   





)

MATCHERS=("gim-lg" "disk-lg" "xfeat")

DATA_DIR="data/C3VD"
RESULTS_DIR="results"
STEP=30
POSTFILTER_STEP=1

mkdir -p "$DATA_DIR"

for video_name in "${!VIDEOS[@]}"; do
    file_id="${VIDEOS[$video_name]}"
    zip_path="${DATA_DIR}/${video_name}.zip"
    extract_path="${DATA_DIR}/${video_name}"


    if [[ -d "$extract_path" ]]; then
        echo "✅ Already extracted: $video_name, skipping download and unzip"
    else
        if [[ -f "$zip_path" ]]; then
            echo "📦 ZIP already downloaded: $zip_path"
        else
            echo "📥 Downloading $video_name..."
            download_gdrive "$file_id" "$zip_path"
        fi

        echo "📦 Unzipping..."
        unzip -q "$zip_path" -d "$DATA_DIR"
        # rm "$zip_path"


    fi

    # 🧹 Clean and keep only every 30th frame and required files
    echo "🧼 Filtering files for $video_name..."
    cd "$extract_path"

    # Remove unwanted folders if they exist
    rm -rf normals occlusion flow
    # === Rename and filter color images ===
    # for img in $(ls *.png | sort -V); do
    #     frame_num=$(basename "$img" .png)
    #     new_name=$(printf "%04d_color.png" "$frame_num")
    #     mv "$img" "$new_name"
        
    #     rm -f "$img"
        
    # done

    # Delete unwanted frames
    for img in *_color.png; do
        frame_num=$(basename "$img" | sed -E 's/^([0-9]+)_color\.png$/\1/')
        if (( 10#$frame_num % STEP != 0 )); then
            rm -f "$img"
        fi
    done

    for depth in *_depth.tiff; do
        frame_num=$(basename "$depth" | sed -E 's/^([0-9]+)_depth\.tiff$/\1/')
        if (( 10#$frame_num % STEP != 0 )); then
            rm -f "$depth"
        fi
    done

    # Remove everything except kept frames, pose.txt, and coverage_mesh.obj
    find . -type f ! \( -name '*_color.png' -o -name '*_depth.tiff' -o -name 'pose.txt' -o -name 'coverage_mesh.obj' \) -delete
    cd - > /dev/null

    echo "🔍 Starting matcher analysis for $video_name..."
    for matcher in "${MATCHERS[@]}"; do
        echo "⚙️  Running analysis: $matcher on $video_name"
        PYTHONPATH=$(pwd) python scripts/matching_analysis.py \
            --video_name "$video_name" \
            --matcher "$matcher" \
            --out_dir "$RESULTS_DIR" \
            --step "$POSTFILTER_STEP"
    done

    # echo "🧹 Cleaning up $video_name files to save space..."
    # rm -rf "$extract_path"

    echo "✅ Done with $video_name"
    echo "-----------------------------"
done


echo "🎉 All videos processed."
