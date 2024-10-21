# ----- user definitions ----- #

PYTHON=/home/luke/pyenv/py38_general/bin/python
SCRIPT=/home/luke/chess/python/assemble_data.py
ARGS=--generate-data
LOG_FOLDER=/home/luke/chess/data/logs

# fen string files, must be compatible with expected format in the python script
FILES=("ficsgamesdb_2021_standard2000_nomovetimes.txt" "ficsgamesdb_2022_standard2000_nomovetimes.txt" "ficsgamesdb_2023_standard2000_nomovetimes.txt")
FILE_NUM=3

# ----- helpful functions ----- #

parseJobs()
{
    # if jobs are colon seperated, eg '1:4', expand this to '1 2 3 4'
    if [[ "$@" == *":"* ]]
    then
        tosearch="$@"
        colonarray=(${tosearch//:/ })
        firstnum=${colonarray[0]}
        endnum=${colonarray[-1]}
        newjobarray=( )
        for ((i=$firstnum;i<=$endnum;i++)); do
            newjobarray+="$i " 
        done
        echo $newjobarray
    else
        # if not colon seperated, return what we were given
        echo $@
    fi
}

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# ----- handle inputs ----- #

PY_ARGS=() # arguments passed directly into python without parsing
LOGGING='Y' # logging enabled by default
NUM_RAND=4069 # default number of positions to generate for

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, increment i
    -j | --jobs ) (( i++ )); jobs=$(parseJobs ${!i}); echo jobs are $jobs ;;
    -s | --stagger ) (( i++ )); STAGGER=${!i}; echo stagger is $STAGGER ;;
    -n | --num-rand ) (( i++ )); NUM_RAND=${!i}; echo num_rand is $NUM_RAND ;;
    # without arguments
    -d | --debug ) LOGGING='N'; echo Debug mode on, terminal logging on ;;
    # everything else passed directly to python
    * ) PY_ARGS+=( ${!i} ) ;;
  esac
done

echo Arguments passed to python script are: ${PY_ARGS[@]}

# ----- main job submission ----- #

# create the log folder if needed
if [ $LOGGING = 'Y' ]
then
    mkdir -p $LOG_FOLDER
fi

# extract the job indicies
ARRAY_INDEXES=("$jobs")

# loop through the jobs we have been assigned
for I in ${ARRAY_INDEXES[@]}
do

    JOB_NAME="data_generation_job_${I}"

    # if we are logging terminal output to a seperate log file
    if [ $LOGGING = 'Y' ]
    then
        # first create the needed file (: is null operator), then direct output to it
        : > $LOG_FOLDER/$JOB_NAME.txt
        exec > $LOG_FOLDER/$JOB_NAME.txt
    fi

    # execute the command in the background
    $FAKETTY $PYTHON $SCRIPT $ARGS \
        --data-file ${FILES[$(($I % $FILE_NUM))]} \
        ${PY_ARGS[@]} \
        --num-rand $NUM_RAND \
        &

    # return output to terminal
    exec > /dev/tty

    echo Submitted job: "$JOB_NAME"

    # for submitting staggered jobs
    IND=$((IND + 1))
    if [ ! -z "$STAGGER" ]
    then
        if [ $(expr $IND % $STAGGER) == "0" ];
        then
            echo -e "Staggering now, waiting for all jobs to finish..."
            wait
            echo -e " ...finished\n"
        fi
    fi

done

echo All jobs submitted

echo Waiting for submitted jobs to complete...
wait 

echo ...finished all jobs