./makeHist.exe sample_signal_v2.root `ls -1v gridpack/Data/zajjQED/ROOTfiles/* | head -200` & > logsignal.txt
./makeHist.exe sample_za_v2.root `ls -1v gridpack/Data/za/ROOTfiles/* | head -80` & > logza.txt
./makeHist.exe sample_zaj_v2.root `ls -1v gridpack/Data/zaj/ROOTfiles/* | head -10` & > logzaj.txt
./makeHist.exe sample_QCD120M_v2.root `ls -1v gridpack/Data/zajjQCD120M/ROOTfiles/* | head -30` & > logQCD120.txt
./makeHist.exe sample_QCD600M_v2.root `ls -1v gridpack/Data/zajjQCD600M/ROOTfiles/* | head -30` & > logQCD600.txt
./makeHist.exe sample_QCD1000M_v2.root `ls -1v gridpack/Data/zajjQCD1000M/ROOTfiles/* | head -30` & > logQCD1000.txt



