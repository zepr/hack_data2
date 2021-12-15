sin_groupama_eau = sin_groupama %>%
  filter(LIB_ALEA=="EXCES D EAU") %>%
  filter(DEPT_PARCELLE %in% c("79","86")) %>%
  filter(GROUPE_CULTURES %in% c("BLE DUR","BLE TENDRE") ) %>%
  mutate(date=as.character(date(DATE_SURVENANCE)),jour=day(DATE_SURVENANCE), mois=month(DATE_SURVENANCE),annee=year(DATE_SURVENANCE)) %>%
  mutate(date=as.numeric(paste0(substr(date,1,4),substr(date,6,7),substr(date,9,10)))) %>%
  mutate(TX_PERTE_RENDEMENT=as.numeric(as.character(gsub(',','.',TX_PERTE_RENDEMENT)))) %>%
  filter(TX_PERTE_RENDEMENT>0.05) %>%
  mutate(sinistre=1)

com_sin= sin_groupama %>%
  filter(DEPT_PARCELLE %in% c("79","86")) %>%
  filter(GROUPE_CULTURES %in% c("BLE DUR","BLE TENDRE") ) %>%
  distinct(COMMUNE) %>% rename(insee=COMMUNE)

 


meteo_sin = histo_meteo %>%
  left_join(sin_groupama_eau %>% select(COMMUNE,sinistre,date),by=c("insee"="COMMUNE","DATE"="date")) %>%
  select(-c(HNEIGEF,GRELE)) %>%
  mutate(RR=as.numeric(gsub(",",".",RR)),
         DRR=as.numeric(gsub(",",".",DRR)),
         intensite=ifelse(DRR==0,0,RR/DRR),
         TN=as.numeric(gsub(",",".",TN)),
         TX=as.numeric(gsub(",",".",TX)),
         TM=as.numeric(gsub(",",".",TM)),
         DG=as.numeric(gsub(",",".",DG)),
         TAMPLI=as.numeric(gsub(",",".",TAMPLI)),
         FFM=as.numeric(gsub(",",".",FFM)),
         FXI=as.numeric(gsub(",",".",FXI)),
         FXY=as.numeric(gsub(",",".",FXY)),
         UM=as.numeric(gsub(",",".",UM)),
         GLOT=as.numeric(gsub(",",".",GLOT)),
         ETPMON=as.numeric(gsub(",",".",ETPMON)),
         sinistre=ifelse(is.na(sinistre)==T,0,sinistre)
         
         
         ) %>% inner_join(com_sin,by="insee")


meteo_sin=meteo_sin[complete.cases(meteo_sin), ]
meteo_sin=meteo_sin %>%
  mutate(mois=substr(DATE,5,6),annee=substr(DATE,1,4),jour=substr(DATE,7,8))%>%
  dplyr::filter(jour !='01') %>%
  dplyr::filter(jour !='02') %>%
  dplyr::filter(jour !='03')


 
# meteo_sin=meteo_sin %>%
#   filter(is.na(RR)==F & is.na(TM)==F & is.na(FFM)==F & is.na(ETPMON)==F)

table(meteo_sin$sinistre)
#Undersampling sur les 0

meteo_sin_0 = meteo_sin %>%
  filter(sinistre==0) %>%
  mutate(rand=runif(nrow(meteo_sin %>% filter(sinistre==0)), 1,100)) %>%
  filter(rand>99) %>%
  select(-c(rand))

meteo_sin_1 = meteo_sin %>%
  filter(sinistre==1)


meteo_sin_under=rbind(meteo_sin_0,meteo_sin_1)
 

prop.table(table(meteo_sin_under$sinistre))

cor(meteo_sin_under[,c(4,5,18)])

meteo_sin_under$sinistre <- factor(meteo_sin_under$sinistre)
# mylogit <- glm(sinistre ~ RR+DRR+TN+TX+TM+DG+TAMPLI+FFM+FXI+FXY+UM+GLOT+ETPMON, data = meteo_sin_under, family = "binomial")
# summary(mylogit)

mylogit <- glm(sinistre ~ RR+FFM+ETPMON+mois, data = meteo_sin_under, family = "binomial")
summary(mylogit)

meteo_sin_under$pdata <- predict(mylogit, newdata = meteo_sin_under, type = "response") 

meteo_sin_under = meteo_sin_under%>%
  mutate(sinistre_pred=ifelse(pdata>0.12,1,0))

table(meteo_sin_under$sinistre,meteo_sin_under$sinistre_pred)

roc_obj <- roc(meteo_sin_under$sinistre, meteo_sin_under$pdata)
auc(roc_obj)




# Rajout du rendement

meteo_sin_under_rend= meteo_sin_under %>%
left_join(sin_groupama_eau %>% select(COMMUNE,TX_PERTE_RENDEMENT,date),by=c("insee"="COMMUNE","DATE"="date"))



rr=meteo_sin_under_rend %>% filter(sinistre==1)

pp=histo_meteo %>%
  filter(insee=="86182")
  

tt= meteo_sin %>%
  filter(insee=="86217")
tt1= sin_groupama %>%
  filter(COMMUNE=="79115")
rr=meteo_sin_under %>% filter(sinistre==1) 
hist(rr$RR)
rr1=meteo_sin_under %>% filter(sinistre==0) 
hist(rr1$RR)
# ## Area under the curve: 0.825
# roc_df <- data.frame(
#   TPR=rev(roc_obj$sensitivities), 
#   FPR=rev(1 - roc_obj$specificities), 
#   labels=roc_obj$response, 
#   scores=roc_obj$predictor)
# 
# roc(meteo_sin_under$sinistre, meteo_sin_under$pdata, smooth=TRUE)
# # this is not identical to
# smooth(roc(meteo_sin_under$sinistre, meteo_sin_under$pdata))
# data(aSAH)
# 
# # Basic example
# roc(aSAH$outcome, aSAH$s100b,
#     levels=c("Good", "Poor"))
# 
# plot(roc(meteo_sin_under$sinistre, meteo_sin_under$pdata, smooth=TRUE))

prop.table(table(meteo_sin_under$sinistre,meteo_sin_under$mois),2)
prop.table(table(meteo_sin_under$sinistre,meteo_sin_under$jour),2)



#Appliquer le modèle

prev_meteo1$pdata <- predict(mylogit, newdata = prev_meteo1, type = "response")

prev_meteo1_= prev_meteo1 %>%
  mutate(annee=as.numeric(substr(Date,1,4))) %>%
  filter(annee>=2022 & annee<=2040) %>%
  filter(station<=10 | station==94 ) %>%
  dplyr::select(Date,station,pdata) %>%
  ungroup() %>%
  mutate(code_insee=
           ifelse(station==1,"79106",
           ifelse(station==2,"79198",
           ifelse(station==3,"79153",
           ifelse(station==4,"79211",
           ifelse(station==5,"79033",
           ifelse(station==6,"79350",
           ifelse(station==7,"79348",
           ifelse(station==8,"79057",
           ifelse(station==9,"79083",
           ifelse(station==10,"79175",
           ifelse(station==94,"79132",""
  )))))))))))) %>%
  select(code_insee,Date,pdata)

names(prev_meteo1_) <- c("CODE_COMMUNE","Date","pred_eau")

write.csv2(prev_meteo1_,file=paste0(Sys.getenv('proj_dir'),"OUTPUT/hackathon_2021/export_prev_alea_eau.csv"))
prev_emma=read.csv(paste0(Sys.getenv('proj_dir'),"OUTPUT/hackathon_2021/prevision_secheresse_to_2040.csv"),sep=";")

prev_meteo1_final = prev_meteo1_ %>%
  mutate(CODE_COMMUNE=as.numeric(CODE_COMMUNE)) %>%
  inner_join(prev_emma,by=c("CODE_COMMUNE","Date"))

names(prev_meteo1_final) <- c("CODE_COMMUNE","Date","pred_eau","pred_sech")
write.csv2(prev_meteo1_final,file=paste0(Sys.getenv('proj_dir'),"OUTPUT/hackathon_2021/export_prev_alea_eau_sech_2040.csv"))

i=1
for (i in 1:max(prev_meteo1$station)) {
  print(i)
  tt= prev_meteo1 %>% filter(station==i)
  plot(tt$pdata,main=paste("station",i))  
}


prev_meteo_pess1$pdata <- predict(mylogit, newdata = prev_meteo_pess1, type = "response")

prev_meteo_pess1= prev_meteo_pess1 %>%
  mutate(annee=substr(Date,1,4)) %>%
  filter(annee=='2023')
i=1
for (i in 1:max(prev_meteo_pess1$station)) {
  print(i)
  tt= prev_meteo_pess1 %>% filter(station==i)
  plot(tt$pdata,main=paste("station",i))  
}

