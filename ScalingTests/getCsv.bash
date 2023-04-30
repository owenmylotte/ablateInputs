for dir in $PWD/*/sbatch/*
do
    echo $dir
    for file in $dir/*.csv
    do
            echo $file
            cp $file $PWD/csv_outputs
    done
done