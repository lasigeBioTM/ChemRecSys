# ChemRecSys
Chemical Compounds Recommender System

This a module for recommending Chemical Compounds. 

There are five algorithms tested:
* ALS (https://implicit.readthedocs.io/en/latest/quickstart.html)
* BPR (https://implicit.readthedocs.io/en/latest/quickstart.html)
* ONTO (Hybrid Semantic Recommender System for Chemical Compounds)
    * ONTO is a Content-Based algorithm, based on the semantic similarity og the Chemical Coumponds in the Chebi ontology

* ALS_ONTO (Hybrid Semantic Recommender System for Chemical Compounds)
* BPR_ONTO (Hybrid Semantic Recommender System for Chemical Compounds)
 


## Requirements:
1) Docker
2) mySQL


## Files:
1. csv with the dataset

    https://drive.google.com/open?id=1AbYgGw7V7KgSLudwxBAbH4yZrwHlBuFG
    (cheRM_20_200.csv)

    [Using Research Literature to Generate Datasets of Implicit Feedback for Recommending Scientific Items](https://ieeexplore.ieee.org/document/8924687)

2. mySQL DB with the compounds similarities (chebi_semantic_sim_cherm_20_200_dump.sql or create at: https://github.com/lasigeBioTM/SemanticSimDBcreator)

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

1) docker build -t "chemrec" .

2) docker run -t -d --name chemrec_container --net=host -v /path/to/data/folder:/mlData -v /path/to/ChemRecSys:/ChemRecSys chemrec

3) docker exec -it chemrec_container bash

4) cd /ChemRecSys/src

5) python main.py

* Note: use config.ini to change the configurations 

The results are saved as CSV files in the folder corresponding to mlData 



