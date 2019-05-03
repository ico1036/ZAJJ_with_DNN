#include <iostream>
using namespace std;

void p_value(){




//double H0=8782.184;
//double H1=8923.139;

double H0=6164;
double H1=H0+137;

double significance=0;
double p_value=0;

double integral=1;
	
	for(int i=0; i<H1; i++){

		integral -= TMath::Poisson(i,H0);

	}

	p_value = integral;
	significance = ROOT::Math::gaussian_quantile_c(p_value,1);
	cout << "#####################################" << endl;
	cout << "p_value: " << p_value << endl;
	cout << "significance: " << significance << endl;
	cout << "#####################################" << endl;


}
