# Example

## Setup 
```
floyd login
floyd init blackhc/projects/idao
```

## Running a job
Then you can use:
```
floyd run --mode job --cpu --data blackhc/datasets/idao_uid_id3_date:data -m "Hello world 1" "python hello_world.py"
```

`idao_uid_id3_date` contains the full dataset sorted by uid, id3 and date (in that order).

`floyd_requirements.txt` is needed to specify additional dependencies for python (tables in this case to make it load hdf files).

A sample run is here https://www.floydhub.com/blackhc/projects/idao/7/

## Dataset

https://www.floydhub.com/blackhc/datasets/idao_uid_id3_date

## Floydhub documentation

https://docs.floydhub.com/
