
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#Script directory is obtained 
pushd "${SCRIPT_DIR}"


##Installing requirements 
sudo apt-get install python3-pip
python3 -m pip install -r requirements.txt


rm -v results/* #Delete old result videos 
sqlite3 db.sqlite3 "delete from entries;" #Delete old database entries 
mkdir "results/rotate"  #Create a folder for rotated videos 

#Run for static.mp4 
REPO_DYNAMIC_RUN=0 python3 -m src & 
pid_1=$! 

sleep 15 #Wait for the program to run for a while and create a holder entry in the database. Otherwise, both programs will have some holder code and clash with each other 

#Run for dynamic.mp4 
REPO_DYNAMIC_RUN=1 python3 -m src &
pid_2=$!

wait $pid_1
wait $pid_2
#Wait for both files to be stabilized by all cascades 

##Some of the videos tend to be vertically flipped for some reason 
#Some weird CV2 bugs as per stack overflow 
#So we individually find the rotated videos and then re-rotate them using ffmpeg 

notif -t "ME 735" -m "Static and Dynamic stabilization completed"
notif -t "ME 735" -m "Please move files to rotate inside results/rotate"
read -p "Press enter to continue"

for file in $(ls results/rotate/*.mp4); do
    ffmpeg -i $file -y -vf "transpose=2,transpose=2" "results/$(basename -- $file)"
done

#Use laser tracker for obtaining the final results 
python3 -m src.LaserTracker 
cp results/*.xlsx $HOME/Desktop/. #Copy the results to the desktop
notif -t "ME 735" -m "Results exported" 
popd #Return to the original directory and proceed as a normal user 