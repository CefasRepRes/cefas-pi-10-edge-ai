#!/bin/bash

# This script is written to be run alongside blobcontainerazcopy.sh
# Whilst the other script downloads the data, for each date, this script classifies it, gets GPS and size info, then deletes the data.


dir="/home/joe/git/rapid-plankton/edge-ai"
cd $dir || exit 1 

temp_dir="/home/joe/Downloads/downloaded"

current_datetime=$(date +'%Y-%m-%d')
downloads_log="downloaded_log_alpha.txt"
processing_log="processed_log_alpha.txt"

# Clear log files
> "$downloads_log"
> "$processing_log"

source env/bin/activate
#source env/scripts/activate


apply_gps_folder() {
      folder="$1"
      bn=$(basename "$folder")
      echo $bn
      python gps.py --folder "$folder" --output /media/joe/47b431b4-fec1-4324-bc2c-632bd1d11f34/gps_exports${folder//\//_}.csv    
}
export -f apply_gps_folder


apply_size_folder() {
      folder="$1"
      bn=$(basename "$folder")
      echo $bn
      python extractor.py --folder "$folder" --output /media/joe/47b431b4-fec1-4324-bc2c-632bd1d11f34/sizes${folder//\//_}.csv
}
export -f apply_size_folder

 
  
# Function to extract folder name from log line
get_folder_name() {
    echo "$1" | awk -F'folder: ' '{print $2}' | cut -d'.' -f1
}

# Create processed file if not exists
touch "$processing_log"

# Initial processing of existing instances
#process_new_instances

# Monitor for new instances
while true; do
    new_instances=$(grep "Download successful" "$downloads_log" | grep -vFxf "$processing_log")
    if [[ -n "$new_instances" ]]; then
	new_instances=$(head -n 1 <<< "$new_instances")
        echo "$new_instances" >> "$processing_log"
        folder=$(get_folder_name "$new_instances")
        echo "Processing folder: $folder"

        # GPS
        find $temp_dir/$folder -type d | xargs -P 12 -I {} bash -c 'apply_gps_folder "$@"' _ {}
        
        # Size...    
        find $temp_dir/$folder -type d | xargs -P 12 -I {} bash -c 'apply_size_folder "$@"' _ {}    
        
        # Classify
        ./classifier.py --folder $temp_dir/$folder --output /media/joe/47b431b4-fec1-4324-bc2c-632bd1d11f34/classifications${folder//\//_}.csv -m 2 -b 100 -r 1        
        
        # Delete
        sleep 1
        rm -rf $temp_dir/$folder
        trash-empty
    fi
    sleep 5
done
