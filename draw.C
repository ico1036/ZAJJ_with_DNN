void draw(){


// --Histogram Parameters
	double XMIN = 0;
	double XMAX = 1000;
	long YMAX = 200	;
	rebin=1;

// --Physics Parameters
	double Lumi					= 150000.0 ;
	
	double xsecSignal			= 0.01291  ;
	double xsecZA				= 7.917    ;
	double xsecZAj				= 2.722    ;
	double xsecZAjjQCD120M		= 0.5274   ;
	double xsecZAjjQCD600M		= 0.02727  ;
	double xsecZAjjQCD1000M	    = 0.008706 ;
	double xsecBKG = xsecZA + xsecZAj + xsecZAjjQCD120M + xsecZAjjQCD600M + xsecZAjjQCD1000M ;
	
	int NgenSignal		= 100000;
	int NgenZA			= 800000;
	int NgenZAj			= 100000;
	int NgenZAjjQCD120M	 = 30000;
	int NgenZAjjQCD600M	 = 30000;
	int NgenZAjjQCD1000M = 30000;
	int NgenBKG = NgenZA+NgenZAj+NgenZAjjQCD120M+NgenZAjjQCD600M+NgenZAjjQCD1000M;


// --Read File/hist
	TFile *fSignal = TFile::Open("sample_signal_v2.root")			;
	TFile *fBKG	   = TFile::Open("sample_main_bkg_v2.root")		;
	//TFile *fZA	   = TFile::Open("sample_za.root")				;
	//TFile *fZAj	   = TFile::Open("sample_zaj.root")				;
	//TFile *fZAjjQCD120M  = TFile::Open("sample_QCD120M.root")	;
	//TFile *fZAjjQCD600M  = TFile::Open("sample_QCD600M.root")	;
	//TFile *fZAjjQCD1000M = TFile::Open("sample_QCD1000M.root")	;

// --Hist list
	//TString histname = "h_ele1PT";     rebin=20; XMIN=0; XMAX=1000; YMAX=10000; TString title_name="PT_{e1}"; 	
    //TString histname = "h_ele2PT";	 rebin=15; XMIN=0; XMAX=300; YMAX=10000;  TString title_name="PT_{e2}";
    //TString histname = "h_ele1Eta"	;rebin=20; XMIN=-3; XMAX=3; YMAX=10000; TString title_name="#eta_{e1}";
    //TString histname = "h_ele2Eta"	;rebin=20; XMIN=-3; XMAX=3; YMAX=10000; TString title_name="#eta_{e2}";
    //TString histname = "h_ele1Phi"	;rebin=30; XMIN=-3.2; XMAX=3.2; YMAX=10000; TString title_name="#phi_{e1}";
    //TString histname = "h_ele2Phi"	;rebin=1; XMIN=-3.15; XMAX=3.15; YMAX=10000; TString title_name="#phi_{e2}";
    //TString histname = "h_phoPT"		; rebin=20; XMIN=0; XMAX=1000; YMAX=10000; TString title_name="PT_{#gamma}";
    //TString histname = "h_phoEta"		; rebin=20; XMIN=-3; XMAX=3; YMAX=10000; TString title_name="#eta_{#gamma}"  ;
    //TString histname = "h_phoPhi"		; rebin=30; XMIN=-3.2; XMAX=3.2; YMAX=10000; TString title_name="#phi_{#gamma}";
    //TString histname = "h_eeM"		; rebin=10; XMIN=60; XMAX=120; YMAX=10000; TString title_name="M_{ee}";
    //TString histname = "h_eeaM"		;rebin=10; XMIN=0; XMAX=2500; YMAX=10000; TString title_name="M_{eea}";
    //TString histname = "h_jet1PT"		; rebin=10; XMIN=0; XMAX=2000; YMAX=10000; TString title_name="PT_{jet1}";
    //TString histname = "h_jet2PT"		; rebin=30; XMIN=0; XMAX=1200; YMAX=10000; TString title_name="PT_{jet2}";
    //TString histname = "h_jet1Eta"	;rebin=10; XMIN=-6; XMAX=6; YMAX=10000; TString title_name="#eta_{jet1}";
    //TString histname = "h_jet2Eta"	;rebin=10; XMIN=-6; XMAX=6; YMAX=10000; TString title_name="#eta_{jet2}";
    //TString histname = "h_jet1Phi"	;rebin=30; XMIN=-3.2; XMAX=3.2; YMAX=10000; TString title_name="#phi_{e2}";
    //TString histname = "h_jet2Phi"	;rebin=30; XMIN=-3.2; XMAX=3.2; YMAX=10000; TString title_name="#phi_{e2}";
    //TString histname = "h1_jjM"		;rebin=15; XMIN=0; XMAX=7000; YMAX=10000; TString title_name="M_{jj}";
    //TString histname = "h1_jdEta"	;	 rebin=20; XMIN=0; XMAX=10; YMAX=10000; TString title_name="#Delta#eta_{jj}";
    //TString histname = "h1_jdPhi"	;	 rebin=20; XMIN=0; XMAX=3.15; YMAX=10000; TString title_name="#Delta#phi_{jj}";
    //TString histname = "h1_ZpVal";     rebin=30; XMIN=0; XMAX=10; YMAX=2000; TString title_name ="ZeppenFeld";
	//TString histname = "h_dRj1l1"; rebin=15; XMIN=0; XMAX=8; YMAX=2000; TString title_name ="$DeltaR_{j1l1}";
	//TString histname = "h_dRj1l2"; rebin=15; XMIN=0; XMAX=10; YMAX=2000; TString title_name ="$DeltaR_{j1l2}";
	//TString histname = "h_dRj2l1"; rebin=15; XMIN=0; XMAX=8; YMAX=2000; TString title_name ="$DeltaR_{j2l1}";
	//TString histname = "h_dRj2l2"; rebin=15; XMIN=0; XMAX=10; YMAX=2000; TString title_name ="$DeltaR_{j2l2}";
    //TString histname = "h_dRjj";	 rebin=15; XMIN=0; XMAX=10; YMAX=2000; TString title_name ="$DeltaR_{jj}";
    TString histname = "h_deltaPhi_ZAjj"; rebin=20; XMIN=0; XMAX=3.2; YMAX=5000; TString title_name ="$Delta#phi_{Z,jj}";



// --Get histogram & Weighting
	TH1F *hSignal	     = (TH1F*)fSignal	   ->Get(histname); hSignal		   ->Scale(xsecSignal	    /NgenSignal		  *Lumi);
	TH1F *hBKG		     = (TH1F*)fBKG		   ->Get(histname); hBKG		   ->Scale(xsecBKG		    /NgenBKG		  *Lumi);
	//TH1F *hZA			 = (TH1F*)fZA		   ->Get(histname); hZA			   ->Scale(xsecZA		    /NgenZA			  *Lumi);
	//TH1F *hZAj			 = (TH1F*)fZAj		   ->Get(histname); hZAj		   ->Scale(xsecZAj		    /NgenZAj		  *Lumi);
	//TH1F *hZAjjQCD120M	 = (TH1F*)fZAjjQCD120M ->Get(histname); hZAjjQCD120M   ->Scale(xsecZAjjQCD120M  /NgenZAjjQCD120M  *Lumi);
	//TH1F *hZAjjQCD600M	 = (TH1F*)fZAjjQCD600M ->Get(histname); hZAjjQCD600M   ->Scale(xsecZAjjQCD600M  /NgenZAjjQCD600M  *Lumi);
	//TH1F *hZAjjQCD1000M	 = (TH1F*)fZAjjQCD1000M->Get(histname); hZAjjQCD1000M  ->Scale(xsecZAjjQCD1000M /NgenZAjjQCD1000M *Lumi);
	




// --Design histogram
	hSignal->SetLineWidth(3); hSignal->SetLineColor(2);
	hBKG->SetLineWidth(3);	  hBKG->SetLineColor(38);	
	hSignal->Rebin(rebin);
	hBKG   ->Rebin(rebin);
	double binwidth= hBKG->GetBinWidth(1);


// --Pad Design
    gStyle->SetOptStat(0);
    gStyle->SetCanvasColor(0);
    gStyle->SetCanvasBorderMode(0);
    gStyle->SetPadBorderMode(0);
    gStyle->SetPadColor(0);
	gStyle->SetFrameBorderMode(0);

	TCanvas* c1 = new TCanvas("c1", "c1", 500, 500);
	TPad *pad1 = new TPad("pad1", "pad1", 0.0, 0.0001, 1.0, 1.0);
		//   pad1->SetBottomMargin(0.01);
		pad1->SetGrid();
		pad1->SetLogy();
		pad1->Draw();
		pad1->cd();
		TH2F *null1 = new TH2F("null1","", 2, XMIN, XMAX, 2, 0.09,YMAX);
		null1->GetYaxis()->SetTitle(Form("Number of events / %3.1f GeV",binwidth));
		null1->GetXaxis()->SetTitle(title_name);
		null1->GetYaxis()->SetTitleOffset(1.8);
		null1->GetXaxis()->SetTitleOffset(1.2);
		null1->GetYaxis()->SetTitleSize(0.03);
		null1->GetYaxis()->SetLabelSize(0.03);
		null1->Draw();
		 hBKG->Draw("same");
		 hSignal->Draw("same");

// --Legend and Latex	
	TLegend *l0 = new TLegend(0.65,0.89,0.90,0.65);
		l0->SetFillStyle(0);
		l0->SetBorderSize(0);
		l0->SetTextSize(0.03);

		TLegendEntry* l01 = l0->AddEntry(hSignal,"Signal"   ,"l"  );    l01->SetTextColor(hSignal->GetLineColor());  
		TLegendEntry* l02 = l0->AddEntry(hBKG,"Background"     ,"l"  ); l02->SetTextColor(hBKG->GetLineColor());
		l0->Draw();

	pad1->cd();
		TLatex latex;
		latex.SetNDC();
		latex.SetTextSize(0.04);
		latex.SetTextAlign(11);
		latex.DrawLatex(0.6,0.91,Form("%.1f fb^{-1} (13 TeV)", Lumi/1000.0));
		
		TString pngname=histname + ".png";
		c1->Print(pngname);

}
