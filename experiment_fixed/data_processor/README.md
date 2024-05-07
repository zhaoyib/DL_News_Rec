#### About data_processor<br>
It is the base part of the whole project.<br>
At the very beginning, I just put all of the processing logic into one simple class, which even not inherited the pytorch Dataloader. Then I restructed the whole code and split it into the following components:<br>
1. BaseDataset.py, it is the BaseDataset which inherited the pytorch Dataset.
2. NewsDataset.py, it inherited the BaseDataset and I added a method called `get_NewsDataset` to get it easily.
3. Dataloader.py, it just inherited the Dataloader of pytorch, I did it for future extensions.
4. Dataloader.gitignore, it is the original code restructed.<br>

In fact, this folder doesn't contain all of the porcessing logic. Part of it has been divided into the utils/load_from_files.py, it is responsible to load the files.