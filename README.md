# ChemRecSys
Chemical Compounds Recommender System

This framework has implemented a Hybrid Semantic Recommender Algorithm for Chemical Compounds. 
It tests several recommender algorithms algorithms:  

* ALS (https://implicit.readthedocs.io/en/latest/quickstart.html)
* BPR (https://implicit.readthedocs.io/en/latest/quickstart.html)
* ONTO (three semantic similarity metrics)
* ALS_ONTO
* BPR_ONTO 

[Hybrid Semantic Recommender System for Chemical Compounds](https://link.springer.com/chapter/10.1007/978-3-030-45442-5_12)
 

## Requirements:
* Python > 3.5
* mySQL
* check requirements.txt


## Input:
1 -  csv with a dataset with the format of <user,item,rating>, where the items are Chemical Compounds in the ChEBI ontology.

    * These files may be found [here](https://drive.google.com/drive/folders/1As_BiqAAcLdUXSY2y49H-XLB87fvinVu?usp=sharing)
        
       - cheRM_20_200.csv is a sample file with 100 items
       - cheRM_20.csv is a more complet file with more than 16000 items 
    
    * Check the creation of these files at:
        [Using Research Literature to Generate Datasets of Implicit Feedback for Recommending Scientific Items](https://ieeexplore.ieee.org/document/8924687)


2 - mySQL DB with the compounds similarities:
We also need a database with the similarities between each compound in the previous csv file. 
This DB may be created by the framework available at https://github.com/lasigeBioTM/SemanticSimDBcreator or downloaded from [here](https://drive.google.com/drive/folders/1As_BiqAAcLdUXSY2y49H-XLB87fvinVu?usp=sharing).

        - chebi_semantic_sim_cherm_20_200_dump.sql has the similarities for cheRM_20_200.csv 
        - cherm_sim_20_dump.sql has the similarities for cheRM_20.csv (more than 131 millions of similarities) 

The DB have three similarities measures to be used by the recommendation algorithms: Resnik, Lin, and Jiang and Conrath (JC)

You may use the following code to dump the mySQL db. 

```
CREATE DATABASE chebi_semantic_sim_cherm_20_200;
USE chebi_semantic_sim_cherm_20_200;

SET FOREIGN_KEY_CHECKS = 0;

DROP TABLE IF EXISTS `similarity`;
CREATE TABLE `similarity` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `comp_1` INT NOT NULL,
  `comp_2` INT NOT NULL,
  `sim_resnik` FLOAT NOT NULL,
  `sim_lin` FLOAT NOT NULL,
  `sim_jc` FLOAT NOT NULL,
  PRIMARY KEY (`id`),
  INDEX sim (`comp_1`,`comp_2`) 
) ENGINE=InnoDB;


SET FOREIGN_KEY_CHECKS = 1;

exit database

mysql -u uname -p chebi_semantic_sim_cherm_20_200 < chebi_semantic_sim_cherm_20_200_dump.sql
```


## Run:

* The first thing to do is to change the config.ini file. 
* Second you should run the file main.py in the src folder. 

* The output will be a set of CSV files with the results for all the algorithms tested with the evaluation metrics:
    * Precision
    * Recall
    * F-Measure
    * False Positive Rate
    * Mean Reciprocal Rank
    * nDCG
    * AUC
    
    
If you wish to use a docker container, here we have an example of how to run:

1) docker build -t "chemrec" .

2) docker run -t -d --name chemrec_container --net=host -v /path/to/data/folder:/mlData -v /path/to/ChemRecSys:/ChemRecSys chemrec

3) docker exec -it chemrec_container bash

4) cd /ChemRecSys/src

5) python main.py
 

The results are saved as CSV files in the folder corresponding to mlData 



