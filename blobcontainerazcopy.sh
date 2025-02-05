#!/bin/bash

# This is the script for downloading data off azure one folder at a time and creating a log of what has been downloaded. As it transpires, azcopy is needed, along with a target folder (date) because azcopy is capable of downloading FAR faster than using the az storage blob copy command. Azcopy is an executable microsoft provide which is capable of optimising the number of concurrent connections for you. You need the right executable for your system (on windows azcopy.exe, on the jetson the program file is just called azcopy).


current_datetime=$(date +'%Y-%m-%d')
downloads_log="downloaded_log_alpha.txt"
processing_log="processed_log_alpha.txt"

# Function to log messages
log() {
    echo "$(date +'%Y-%m-%d %H:%M:%S') - $1" >> "$downloads_log"
}

# Function to retry a command up to a specified number of times
retry_command() {
    local retries=$1
    shift
    local count=0
    until "$@" || [ $count -eq $retries ]; do
        log "Command failed. Retrying..."
        ((count++))
        sleep 1
    done
}


storage_account="citprodc8603uksa"
container_name="databox-2483495b-0221-43b8-584a-444434c92e9c" # Do not use big-cactus, this is unsigned integer format so will break classifier.py 
connection_string=$(</home/joe/Documents/connstr.txt) # this is incorrect?
sas_token=$(</home/joe/Documents/sas.txt) 

# Retrieve the list of folder names
folder_list=$(retry_command 3 az storage blob list --connection-string "$connection_string" --container-name "$container_name" --delimiter '/' --output json | jq -r '.[].name | select(test("/")) | split ("/")[0]' | sort -u)
folder_list="2023-10-12 2023-10-13 2023-10-14 2023-10-15 2023-10-16 2023-10-17 2023-10-18 2023-10-19 2023-10-20 2023-10-21 2023-10-22 2023-10-23 2023-10-24 2023-10-25 2023-10-26 2023-10-27 2023-10-28 2023-10-29 2023-10-30 2023-10-31" # May be useful if your computer restarts part way through...  -_-

cd '/home/joe/git/rapid-plankton/edge-ai' || exit 1
#cd ~/git/rapid-plankton/edge-ai || exit 1


#create_temp_dir() {
#    local temp_dir=$(mktemp -d)
#    echo "$temp_dir"
#}
# \/ not using this for now \/
#    temp_dir=$(create_temp_dir)


# Clear log files
> "$downloads_log"
> "$processing_log"

for folder_name in $folder_list; do
log "Downloading folder: $folder_name"
# Iterate over each minute in the day (valid minutes from '0000' to '2359')
for hour in {00..23}; do
for minute in {00..59}; do
    #clear the azcopy folder
    rm -rf /home/joe/.azcopy
    trash-empty
    
    for waititerations in {0..10}; do
        # Count the number of "Download successful" lines in both log files and check if we are behind
        downloads_count=$(grep -c "Download successful" "$downloads_log")
        processing_count=$(grep -c "Download successful" "$processing_log")
        unprocessed_count=$((downloads_count - processing_count))

        while [ "$unprocessed_count" -gt 2 ]
        do
          echo "More than 2 folders are waiting to be processed. Pausing the download script for 1 minute."
          sleep 60
        done
    done

    minute_folder="${folder_name}/${hour}${minute}"
    echo $minute_folder
    temp_dir="/home/joe/Downloads/downloaded/${folder_name}"
    mkdir -p "$temp_dir"
    log "Copying contents of folder: $minute_folder"
    export AZCOPY_CRED_TYPE="Anonymous"
    export AZCOPY_CONCURRENCY_VALUE="AUTO"
    azcopy_output=$(./azcopy copy "https://$storage_account.blob.core.windows.net/$container_name/$minute_folder/$sas_token" "$temp_dir" --overwrite=true --check-md5 FailIfDifferent --from-to=BlobLocal --recursive --log-level=INFO 2>&1)
    azcopy_exit_code=$?
    echo $azcopy_exit_code
    export AZCOPY_CRED_TYPE=""
    export AZCOPY_CONCURRENCY_VALUE=""
    if [ $azcopy_exit_code == 0 ] && [ -d "$temp_dir" ] && [ "$(ls -A "/home/joe/Downloads/downloaded/${minute_folder}")" ]; then
        log "Download successful for folder: $minute_folder."
    else
        log "Download failed for folder: $minute_folder."
    fi
done
done
done


