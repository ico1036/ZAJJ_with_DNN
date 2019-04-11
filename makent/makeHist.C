#include <iostream>
#include "TClonesArray.h"
#include "TFile.h"
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"
#include "/x5/cms/jwkim/Delphes-3.4.1/classes/DelphesClasses.h"

using namespace std;

int main(int argc, char** argv){

	 TFile* outFile = new TFile(argv[1],"recreate");
     TChain* inChain = new TChain("Delphes");

     int    nttype = atoi(argv[2]) ; // singla:1 BKG: 0
     double ntxsec = atof(argv[3]) ;
     double ntgenN = atof(argv[4]) ;

    for(int iFile = 5; iFile<argc; iFile++) {
        std::cout << "### InFile " << iFile-1 << " " << argv[iFile] << std::endl;
        inChain->Add(argv[iFile]);
    }



	TClonesArray* eleTCA = new TClonesArray("Electron");inChain->SetBranchAddress("Electron",&eleTCA);
    TClonesArray *phoTCA = new TClonesArray("Photon"); inChain->SetBranchAddress("Photon",&phoTCA);
    TClonesArray *jetTCA = new TClonesArray("Jet"); inChain->SetBranchAddress("Jet",&jetTCA);

    TClonesArray *eleSelTCA =  new TClonesArray("Electron");
    TClonesArray *phoSelTCA =  new TClonesArray("Photon");
    TClonesArray *jetSelTCA =  new TClonesArray("Jet");

	TH1F *h_ele1PT  = new TH1F("h_ele1PT","h_ele1PT",1000,0,1000);	
	TH1F *h_ele2PT  = new TH1F("h_ele2PT","h_ele2PT",1000,0,1000);	
	TH1F *h_ele1Eta = new TH1F("h_ele1Eta","h_ele1Eta",1000,-10,10);
	TH1F *h_ele2Eta = new TH1F("h_ele2Eta","h_ele2Eta",1000,-10,10);
	TH1F *h_ele1Phi = new TH1F("h_ele1Phi","h_ele1Phi",1000,-3.15,3.15);
	TH1F *h_ele2Phi = new TH1F("h_ele2Phi","h_ele2Phi",1000,-3.15,3.15);

	TH1F *h_phoPT	= new TH1F("h_phoPT","h_phoPT",1000,0,1000);
	TH1F *h_phoEta	= new TH1F("h_phoEta","h_phoEta",1000,-10,10);
	TH1F *h_phoPhi	= new TH1F("h_phoPhi","h_phoPhi",1000,-3.15,3.15);
	
	TH1F *h_Njet	= new TH1F("h_Njet","h_Njet",15,0,15);
	TH1F *h_jet1PT  = new TH1F("h_jet1PT","h_jet1PT",1000,0,7000);
	TH1F *h_jet2PT  = new TH1F("h_jet2PT","h_jet2PT",1000,0,1000);
	TH1F *h_jet1Eta = new TH1F("h_jet1Eta","h_jet1Eta",1000,-10,10);
	TH1F *h_jet2Eta = new TH1F("h_jet2Eta","h_jet2Eta",1000,-10,10);
	TH1F *h_jet1Phi = new TH1F("h_jet1Phi","h_jet1Phi",1000,-3.15,3.15);
	TH1F *h_jet2Phi = new TH1F("h_jet2Phi","h_jet2Phi",1000,-3.15,3.15);

	TH1F *h_eeM	   = new TH1F("h_eeM","h_eeM",1000,0,200);
	TH1F *h_eeaM   = new TH1F("h_eeaM","h_eeaM",1000,0,7000);
	TH1F * h_jjM   = new TH1F("h1_jjM","h1_jjM",1000,0,7000);
    TH1F * h_jdEta = new TH1F("h1_jdEta","h1_jdEta",1000,0,10);
    TH1F * h_jdPhi = new TH1F("h1_jdPhi","h1_jdPhi",1000,0,TMath::Pi());
    TH1F * h_ZpVar = new TH1F("h1_ZpVal","Zppenfeld",1000,0,10);

	TH1F* h_dRj1l1	= new TH1F("h_dRj1l1","h_dRj1l1",1000,0,15);
	TH1F* h_dRj1l2	= new TH1F("h_dRj1l2","h_dRj1l2",1000,0,15);
	TH1F* h_dRj2l1	= new TH1F("h_dRj2l1","h_dRj2l1",1000,0,15);
	TH1F* h_dRj2l2	= new TH1F("h_dRj2l2","h_dRj2l2",1000,0,15);
	TH1F* h_dRjj		    = new TH1F("h_dRjj"	,"h_dRjj",1000,0,15);
	TH1F* h_deltaPhi_ZAjj	= new TH1F("h_deltaPhi_ZAjj","h_deltaPhi_ZAjj",1000,0,4);




// --Event number parameters
	int totalEvt = (int)inChain->GetEntries();
	int per99 = totalEvt/99;
	int per100 = 0;
	int step0num =0;
	int step1num =0;
    int step2num =0;
    int step3num =0;
	double VBFnum=0;



// --Tree for making ntuples
	TTree* outTree = new TTree("tree","tree");
	outTree->Branch("nttype", &nttype);

	double ntele1PT		;
	double ntele2PT		;
	double ntele1Eta		;
	double ntele2Eta		;
	double ntele1Phi		;
	double ntele2Phi		;
	double ntphoPT		;
	double ntphoEta		;
	double ntphoPhi		;
	double ntjet1PT		;
	double ntjet2PT		;
	double ntjet1Eta		;	 
	double ntjet2Eta		;
	double ntjet1Phi		;
	double ntjet2Phi		;
	double nteeM			;
	double nteeaM			;
	double ntjjM			;
	double ntjdEta			;
	double ntjdPhi			;
	double ntZpVar			;
	double ntdRj1l1		;
	double ntdRj1l2		;
	double ntdRj2l1		;
	double ntdRj2l2		;
	double ntdRjj			;
	double ntdeltaPhi_ZAjj ;
	
	outTree->Branch("ntele1PT",&ntele1PT)		;
	outTree->Branch("ntele2PT",&ntele2PT)		;
	outTree->Branch("ntele1Eta",&ntele1Eta)	;
	outTree->Branch("ntele2Eta",&ntele2Eta)	;
	outTree->Branch("ntele1Phi",&ntele1Phi)	;
	outTree->Branch("ntele2Phi",&ntele2Phi)	;
	outTree->Branch("ntphoPT",&ntphoPT)		;
	outTree->Branch("ntphoEta",&ntphoEta)		;
	outTree->Branch("ntphoPhi",&ntphoPhi)		;
	outTree->Branch("ntjet1PT",&ntjet1PT)		;
	outTree->Branch("ntjet2PT",&ntjet2PT)		;
	outTree->Branch("ntjet1Eta",&ntjet1Eta)	; 
	outTree->Branch("ntjet2Eta",&ntjet2Eta)	;
	outTree->Branch("ntjet1Phi",&ntjet1Phi)	;
	outTree->Branch("ntjet2Phi",&ntjet2Phi)	;
	outTree->Branch("nteeM",&nteeM)			;
	outTree->Branch("nteeaM",&nteeaM)			;
	outTree->Branch("ntjjM",&ntjjM)			;
	outTree->Branch("ntjdEta",&ntjdEta)		;
	outTree->Branch("ntjdPhi",&ntjdPhi)		;
	outTree->Branch("ntZpVar",&ntZpVar)		;
	outTree->Branch("ntdRj1l1",&ntdRj1l1)		;
	outTree->Branch("ntdRj1l2",&ntdRj1l2)		;
	outTree->Branch("ntdRj2l1",&ntdRj2l1)		;
	outTree->Branch("ntdRj2l2",&ntdRj2l2)		;
	outTree->Branch("ntdRjj",&ntdRjj)			;
	outTree->Branch("ntdeltaPhi_ZAjj",&ntdeltaPhi_ZAjj);
	outTree->Branch("nttype",&nttype);
	outTree->Branch("ntxsec",&ntxsec);
	outTree->Branch("ntgenN",&ntgenN);





	// ---EventLoop start
	for(int eventLoop=0; eventLoop < totalEvt; eventLoop++) {
		inChain->GetEntry(eventLoop);
		if((eventLoop%per99) == 0) std::cout << "Running " << per100++ << " %" << std::endl;

		eleSelTCA->Clear("C");
        phoSelTCA->Clear("C");
        jetSelTCA->Clear("C");
			
		step0num++;

/////////////////////--Electron Selection 
		// ---Electron Loop start
		for(int eleLoop=0; eleLoop<eleTCA->GetEntries(); eleLoop++){
            Electron *elePtr = (Electron*)eleTCA->At(eleLoop);

			if (elePtr->PT < 25)            continue;
			if(TMath::Abs(elePtr->Eta)>2.5) continue;

			new ((*eleSelTCA)[(int)eleSelTCA->GetEntries()]) Electron(*elePtr);


		}	 // -- Elctron Loop end
		
		if(eleSelTCA->GetEntries() != 2) continue;
		Electron* elePtr1 = (Electron*)eleSelTCA->At(0);
		Electron* elePtr2 = (Electron*)eleSelTCA->At(1);
		if(elePtr1->Charge * elePtr2->Charge == 1 ) continue;
		 step1num++;  


		
/////////////////////--Photon Selection 
	// ---Photon Loop start	
	for(int phoLoop=0; phoLoop<phoTCA->GetEntries(); phoLoop++){
			Photon *phoPtr = (Photon*)phoTCA->At(phoLoop);

			if(phoPtr->PT <25) continue;
			if(TMath::Abs(phoPtr->Eta)>2.5) continue;
			
			double dPhi1 = (phoPtr->Phi)-(elePtr1->Phi);
			double dPhi2 = (phoPtr->Phi)-(elePtr2->Phi);
			double deltaPhi1 = ( dPhi1 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi1) : fabs(dPhi1);
			double deltaPhi2 = ( dPhi2 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi2) : fabs(dPhi2);
			double dEta1 = fabs(phoPtr->Eta - elePtr1->Eta);
			double dEta2 = fabs(phoPtr->Eta - elePtr2->Eta);
			double dR1 = TMath::Sqrt(deltaPhi1*deltaPhi1 + dEta1*dEta1);
			double dR2 = TMath::Sqrt(deltaPhi2*deltaPhi2 + dEta2*dEta2);
			if(dR1 <0.5 || dR2 < 0.5) continue;

		new ((*phoSelTCA)[(int)phoSelTCA->GetEntries()]) Photon(*phoPtr);


		} // Photon Loop end
		
		if(phoSelTCA->GetEntries()!=1 )  continue;


	    Photon* phoPtr = (Photon*)phoSelTCA->At(0);
		
		
/////////////////////--Z mass window
	TLorentzVector ele1Vec = elePtr1->P4();
   	TLorentzVector ele2Vec = elePtr2->P4();
   	TLorentzVector eeVec = ele1Vec + ele2Vec;
	if(eeVec.M() < 60 || eeVec.M() > 120) continue;
	
	    step2num++;

	

		// if(jetTCA->GetEntries() != 0 ) {
		// Jet* jetPtr = (Jet*)jetTCA->At(0);
		// h1_jetPT->Fill(jetPtr->PT);
		// }
		// h1_Njet->Fill(jetTCA->GetEntries());
	
/////////////////////--Jet Selection 
	// ---Jet Loop start
	for(int jetLoop=0; jetLoop<jetTCA->GetEntries(); jetLoop++){
        Jet *jetPtr = (Jet*)jetTCA->At(jetLoop);

        if(TMath::Abs(jetPtr->Eta)>4.7) continue;
        if(jetPtr->PT < 30) continue;
			

			double dPhi = (phoPtr->Phi)-(jetPtr->Phi);
			double deltaPhi = ( dPhi > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi) : fabs(dPhi);
			double dEta = fabs(phoPtr->Eta - jetPtr->Eta);
			double dR = TMath::Sqrt(deltaPhi*deltaPhi + dEta*dEta);
			if(dR <0.5) continue;


	
	new ((*jetSelTCA)[(int)jetSelTCA->GetEntries()]) Jet(*jetPtr);


    } // Jet Loop End
    
	if(jetSelTCA->GetEntries() < 2) continue;
    step3num++;  

	

    Jet* jetPtr1 = (Jet*)jetSelTCA->At(0);
    Jet* jetPtr2 = (Jet*)jetSelTCA->At(1);
	
	TLorentzVector phoVec = phoPtr->P4();
	TLorentzVector eeaVec = ele1Vec + ele2Vec+phoVec;
	TLorentzVector jet1Vec = jetPtr1->P4();
    TLorentzVector jet2Vec = jetPtr2->P4();
    TLorentzVector jjVec = jet1Vec + jet2Vec;
	
	// Di-jet mass
	double jjM =jjVec.M();
	
	// Delta Phi "dleta pht needs correction" 
	double dEta = fabs((jetPtr2->Eta) - (jetPtr1->Eta));
    double dPhi = fabs((jetPtr2->Phi) - (jetPtr1->Phi));
    double deltaPhi = ( dPhi > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi) : fabs(dPhi); // Correction
    
	
	double rapZA = eeaVec.Rapidity();
    double rapJ1 = jet1Vec.Rapidity();
    double rapJ2 = jet2Vec.Rapidity();
    
	// Zeppenfeld variable
	double zepp = fabs(rapZA - (rapJ1 + rapJ2) / 2.0);
	// dR of two jets
	double dRjj = TMath::Sqrt(deltaPhi*deltaPhi + dEta*dEta);
	
	// Delta Eta,Delta Phi, and dR  of jets,leptons
	double dEta_j1l1 = fabs((jetPtr1->Eta)-(elePtr1->Eta));
	double dEta_j1l2 = fabs((jetPtr1->Eta)-(elePtr2->Eta));
	double dEta_j2l1 = fabs((jetPtr2->Eta)-(elePtr1->Eta));
	double dEta_j2l2 = fabs((jetPtr2->Eta)-(elePtr2->Eta));
	
	double dPhi_j1l1 = fabs((jetPtr1->Phi)-(elePtr1->Phi));
	double dPhi_j1l2 = fabs((jetPtr1->Phi)-(elePtr2->Phi));
	double dPhi_j2l1 = fabs((jetPtr2->Phi)-(elePtr1->Phi));
	double dPhi_j2l2 = fabs((jetPtr2->Phi)-(elePtr2->Phi));
	
    double deltaPhi_j1l1 = ( dPhi_j1l1 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi_j1l1) : fabs(dPhi_j1l1);
    double deltaPhi_j1l2 = ( dPhi_j1l2 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi_j1l2) : fabs(dPhi_j1l2);
    double deltaPhi_j2l1 = ( dPhi_j2l1 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi_j2l1) : fabs(dPhi_j2l1);
    double deltaPhi_j2l2 = ( dPhi_j2l2 > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhi_j2l2) : fabs(dPhi_j2l2);
	
	double dRj1l1 = TMath::Sqrt(deltaPhi_j1l1*deltaPhi_j1l1 + dEta_j1l1*dEta_j1l1);
	double dRj1l2 = TMath::Sqrt(deltaPhi_j1l2*deltaPhi_j1l2 + dEta_j1l2*dEta_j1l2);
	double dRj2l1 = TMath::Sqrt(deltaPhi_j2l1*deltaPhi_j2l1 + dEta_j2l1*dEta_j2l1);
	double dRj2l2 = TMath::Sqrt(deltaPhi_j2l2*deltaPhi_j2l2 + dEta_j2l2*dEta_j2l2);
	
	// Deltaphi of Z and jets
	double Phijj = jjVec.Phi();
	double PhiZA = eeaVec.Phi();
	double dPhiZAjj = fabs(Phijj-PhiZA);
	double deltaPhi_ZAjj = ( dPhiZAjj > TMath::Pi() ) ? fabs(TMath::TwoPi() - dPhiZAjj) : fabs(dPhiZAjj);
	

		h_ele1PT->Fill(elePtr1->PT);
    	h_ele2PT->Fill(elePtr2->PT);
    	h_ele1Eta->Fill(elePtr1->Eta);
    	h_ele2Eta->Fill(elePtr2->Eta);
    	h_ele1Phi->Fill(elePtr1->Phi);
    	h_ele2Phi->Fill(elePtr2->Phi);
    	
    	
		h_phoPT->Fill(phoPtr->PT); 
		h_phoEta->Fill(phoPtr->Eta);   
		h_phoPhi->Fill(phoPtr->Phi);   
		
		
		h_eeM->Fill(eeVec.M());
		h_eeaM->Fill(eeaVec.M());
		
		h_Njet->Fill(jetSelTCA->GetEntries());
		h_jet1PT->Fill(jetPtr1->PT);
    	h_jet2PT->Fill(jetPtr2->PT);
    	h_jet1Eta->Fill(jetPtr1->Eta);
    	h_jet2Eta->Fill(jetPtr2->Eta);
    	h_jet1Phi->Fill(jetPtr1->Phi);
    	h_jet2Phi->Fill(jetPtr2->Phi);
		
		h_jjM->Fill(jjM);
		h_jdEta->Fill(dEta);
		h_jdPhi->Fill(deltaPhi);
		h_ZpVar->Fill(zepp);

		h_dRj1l1->Fill(dRj1l1);
		h_dRj1l2->Fill(dRj1l2);
		h_dRj2l1->Fill(dRj2l1);
		h_dRj2l2->Fill(dRj2l2);
		h_dRjj		   ->Fill(dRjj);
		h_deltaPhi_ZAjj->Fill(deltaPhi_ZAjj);
	
		ntele1PT  =	elePtr1->PT	;
		ntele2PT  =	elePtr2->PT	;
		ntele1Eta =	elePtr1->Eta	;
		ntele2Eta =	elePtr2->Eta	;
		ntele1Phi =	elePtr1->Phi	;
		ntele2Phi =	elePtr2->Phi	;
		ntphoPT	  = phoPtr->PT		;
		ntphoEta  = phoPtr->Eta		;
		ntphoPhi  = phoPtr->Phi		;
		nteeM	  = eeVec.M()		;
		nteeaM    = eeaVec.M()		;
		ntjet1PT  = jetPtr1->PT	;
		ntjet2PT  = jetPtr2->PT	;
		ntjet1Eta = jetPtr1->Eta	;
		ntjet2Eta = jetPtr2->Eta	;
		ntjet1Phi = jetPtr1->Phi	;
		ntjet2Phi = jetPtr2->Phi	;
		ntjjM	  = jjM				;
		ntjdEta	  = dEta			;
		ntjdPhi   = deltaPhi		;
		ntZpVar   = zepp			;
		ntdRj1l1  = dRj1l1			;
		ntdRj1l2  = dRj1l2			;
		ntdRj2l1  = dRj2l1			;
		ntdRj2l2  = dRj2l2			;
		ntdRjj	  = dRjj			;
		ntdeltaPhi_ZAjj = deltaPhi_ZAjj ;
	
		outTree->Fill();
	} //--Event Loop end


	cout << "step0 :" << step0num << endl;
	cout << "step1 :" << step1num << endl;
    cout << "step2 :" << step2num << endl;
    cout << "step3 :" << step3num << endl;

	outFile->Write();
	return 0;

}
