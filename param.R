packrat::off()
setwd(dir = paste0(Sys.getenv('proj_dir'),"STUDIES/hackathon_2021"))

packrat::on() 


library(dplyr)
library(lubridate)
library(pROC)
# library(terra)
# library(geodata)
# library(sf)
# library(RTriangle)

# histo_meteo=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/histo_meteo_dpt_86_79.txt"),sep = ";")
stations_meteo=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/metadata_stations_meteo_86_79.txt"),sep = ";")

histo_meteo=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/base_finale_meteo_V2.csv"),sep = ";")

sin_groupama=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/OUTPUT/bdd_hackathon.csv"),sep = ";")

prev_meteo=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/Future_2022-2080_Dept79.csv"),sep = ";") 

prev_meteo_pess=read.csv(paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/Future_2022-2080_Dept79_Rcp85.csv"),sep = ";") 

prev_meteo_pess1= prev_meteo_pess %>%
  group_by(Latitude,Longitude) %>%
  mutate(station = cur_group_id()) %>%
  mutate(mois=substr(Date,6,7)) %>%
  select(station,RR,FFM,ETPMON,mois,Date) 

prev_meteo1= prev_meteo %>%
  group_by(Latitude,Longitude) %>%
  mutate(station = cur_group_id()) %>%
  mutate(mois=substr(Date,6,7)) %>%
  select(station,RR,FFM,ETPP,mois,Date) %>%
  rename(ETPMON=ETPP)

# write.csv2(prev_meteo1,file=paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/Future_2022-2080_Dept79_idstations.csv"))
# write.csv2(prev_meteo_pess1,file=paste0(Sys.getenv('filer_dir'),"2021_hackathon_varenne_eau/Future_2022-2080_Dept79_Rcp85_idstations.csv"))

