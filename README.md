# Clash Royale Graph Neural Network 

## Objective
A Graph neural network-based system for recommending 2 additional cards given 6 input cards in Clash Royale, trained on battle data from top players.

## Project Structure

ClashRoyalGNN/
├── config/
│   └── config.yaml          
├── data/
│   ├── 01-raw/              
│   ├── 02-preprocessed/     
│   ├── 03-features/         
│   └── 04-predictions/      
├── entrypoint/
│   ├── train.py             
│   └── inference.py         
├── notebooks/
│   ├── Baseline.ipynb       
│   └── EDA.ipynb            
├── src/
│   ├── data_collection/
│   │   ├── cr_api.py       
│   │   └── data_fetcher.py  
│   ├── pipelines/
│   │   ├── __init__.py      
│   │   ├── feature_eng_pipeline.py  
│   │   ├── inference_pipeline.py    
│   │   └── training_pipeline.py     
│   └── utils.py             
├── tests/
│   ├── __init__.py          
│   └── test_training.py     
├── .gitignore               
├── README.md                
└── requirements.txt         

## Data required (Modify the data_collection steps)

Get the data from the Cars: https://api.clashroyale.com/v1/cards 
A successful response (200) will retrieve the following JSON: 
Items{
items	ItemList[Item{
iconUrls	{
}
name	JsonLocalizedName{
}
id	integer
rarity	string
Enum:
[ COMMON, RARE, EPIC, LEGENDARY, CHAMPION ]
maxLevel	integer
elixirCost	integer
maxEvolutionLevel	integer
}]
supportItems	ItemList[Item{
iconUrls	{
}
name	JsonLocalizedName{
}
id	integer
rarity	string
Enum:
[ COMMON, RARE, EPIC, LEGENDARY, CHAMPION ]
maxLevel	integer
elixirCost	integer
maxEvolutionLevel	integer
}]
}


Get all the members from  the clans we can: https://api.clashroyale.com/v1/clans
This API has a parameter called "minScore": Filter by minimum amount of clan score. Need to define a minScore to filter clans. 
A successful response will retrive the following JSON: 
ClanList[Clan{
memberList	ClanMemberList[ClanMember{
arena	Arena{...}
clanChestPoints	integer
lastSeen	string
tag	string
name	string
role	string
Enum:
Array [ 5 ]
expLevel	integer
trophies	integer
clanRank	integer
previousClanRank	integer
donations	integer
donationsReceived	integer
}]
tag	string
clanChestStatus	string
Enum:
[ INACTIVE, ACTIVE, COMPLETED, UNKNOWN ]
clanChestLevel	integer
requiredTrophies	integer
donationsPerWeek	integer
badgeId	integer
clanScore	integer
clanChestMaxLevel	integer
clanWarTrophies	integer
name	string
location	Location{...}
type	string
Enum:
Array [ 3 ]
members	integer
description	string
clanChestPoints	integer
badgeUrls	{
}
}]
