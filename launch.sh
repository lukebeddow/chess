# ----- user definitions ----- #

PYTHON=/home/luke/pyenv/py38_general/bin/python
SCRIPT_GEN_DATA=/home/luke/chess/python/assemble_data.py
SCRIPT_TRAIN=/home/luke/chess/python/train_nn_evaluator.py
LOG_FOLDER=/home/luke/chess/logs

# fen string files for data generation, must be compatible with expected format in the python script
# FILES=("ficsgamesdb_2021_standard2000_nomovetimes.txt" "ficsgamesdb_2022_standard2000_nomovetimes.txt" "ficsgamesdb_2023_standard2000_nomovetimes.txt")
# FILE_NUM=3
FILES=("ficsgamesdb_202303_blitz_nomovetimes.txt" "ficsgamesdb_202303_blitz_nomovetimes.txt")
FILE_NUM=2

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

autoGetTimestamp()
{
    # get the name of the most recent training log file
    recent_run=$(cd $LOG_FOLDER && ls -1t | head -1)
    echo Auto-detected most recent_run is: $recent_run

    # split it by underscore into four variables
    IFS="_" read v1 v2 v3 v4 <<< "$recent_run"

    # reconstruct and save the timestamp
    under="_"
    TIMESTAMP="$v2$under$v3"
    echo Auto-detected timestamp is: $TIMESTAMP
}

print_table()
{
    # print a training recap table
    $FAKETTY $PYTHON $SCRIPT_TRAIN \
        --timestamp $TIMESTAMP \
        --print-table \
        ${PY_ARGS[@]}

    echo "Finished"
    exit
}

# from: https://stackoverflow.com/questions/1401002/how-to-trick-an-application-into-thinking-its-stdout-is-a-terminal-not-a-pipe
faketty() {
    script -qfc "$(printf "%q " "$@")" /dev/null
}

# ----- handle inputs ----- #

TIMESTAMP="$(date +%d-%m-%y_%H-%M)"
PY_ARGS=() # arguments passed directly into python without parsing
LOGGING='Y' # logging enabled by default
PRINTTABLE='N' # print a table
GEN_DATA='N' # not a generate data job by default
NUM_RAND=4096 # default number of positions to generate for

# loop through input args, look for script specific arguments
for (( i = 1; i <= "$#"; i++ ));
do
  case ${!i} in
    # with arguments, increment i
    -j | --jobs ) (( i++ )); jobs=$(parseJobs ${!i}); echo jobs are $jobs ;;
    -t | --timestamp ) (( i++ )); TIMESTAMP=${!i}; echo Timestamp set to $TIMESTAMP ;;
    -s | --stagger ) (( i++ )); STAGGER=${!i}; echo stagger is $STAGGER ;;
    -n | --num-rand ) (( i++ )); NUM_RAND=${!i}; echo num_rand is $NUM_RAND ;;
    # without arguments
    -d | --debug ) LOGGING='N'; echo Debug mode on, terminal logging on ;;
    -g | --generate-data ) GEN_DATA='Y'; echo generate data job selected ;;
    -a | --auto-timestamp ) autoGetTimestamp ;;
    --print-table ) PRINTTABLE='Y'; echo Preparing to print a results table ;;
    # everything else passed directly to python
    * ) PY_ARGS+=( ${!i} ) ;;
  esac
done

echo Arguments passed to python script are: ${PY_ARGS[@]}

# ----- main job submission ----- #

# print a training results table
if [ $PRINTTABLE = 'Y' ]
then
    print_table
fi

# create the log folder if needed
if [ $LOGGING = 'Y' ]
then
    mkdir -p $LOG_FOLDER
fi

# extract the job indicies
ARRAY_INDEXES=("$jobs")

# wrapper to catch ctrl+c and kill all background processes
trap 'trap - SIGINT && kill 0' SIGINT

# loop through the jobs we have been assigned
for I in ${ARRAY_INDEXES[@]}
do

    if [ $GEN_DATA = 'Y' ]
    then
        PRINTTABLE='N'
        JOB_NAME="data_generation_job_${I}"
    else
        PRINTTABLE='Y'
        JOB_NAME="run_${TIMESTAMP}_A${I}"
    fi
    

    # if we are logging terminal output to a seperate log file
    if [ $LOGGING = 'Y' ]
    then
        # first create the needed file (: is null operator), then direct output to it
        : > $LOG_FOLDER/$JOB_NAME.txt
        exec > $LOG_FOLDER/$JOB_NAME.txt
    fi

    if [ $GEN_DATA = 'Y' ]
    then
        # execute data generation command in the background
        $FAKETTY $PYTHON $SCRIPT_GEN_DATA --generate-data \
            --job $I \
            --data-file ${FILES[$(($I % $FILE_NUM))]} \
            ${PY_ARGS[@]} \
            --num-rand $NUM_RAND \
            &
    else
        # run a learning command in the background
        $FAKETTY $PYTHON $SCRIPT_TRAIN \
            --job $I \
            --timestamp $TIMESTAMP \
            ${PY_ARGS[@]} \
            &
    fi

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

# print a training results table
if [ $PRINTTABLE = 'Y' ]
then
    echo "Now printing a table of training results"
    print_table
fi