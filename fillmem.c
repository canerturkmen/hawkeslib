#include <stdio.h>
#include <stdlib.h>

#define MAX_ARRAY_SIZE 37 * 1000 * 1000 // fix this! waste so much memory!
#define DATA_FILENAME "/home/caner/res/hawkes_em/data/Master_Data_Full.csv"

typedef struct  {
	unsigned long *times;
	unsigned short int *codes;
	unsigned int *epar;
	int N;
	short int maxcode;
} HawkesData;

void get_earliest_parent(unsigned long *times, int N, int tmax, int *epar){

	int ptr1 = N-1;
	int ptr2 = N-1;

	while (ptr1 > 0) {
		if (times[ptr1] - times[ptr2] > tmax){
			epar[ptr1] = ptr2 + 1;
			ptr1--;
		}
		else if (ptr2 == 0){
			epar[ptr1] = ptr2;
			ptr1--;
		}
		else ptr2--;
	}

}

int initialize(unsigned long *times, unsigned short int *codes, short int *maxcode_out){

	FILE* stream = fopen(DATA_FILENAME, "r");
	char line[1024];

	int i = 0;
	short int _mcode = 0;

	unsigned long ttime;
	unsigned short int code;

	while (fgets(line, 1024, stream)){

		sscanf(line, "%hu,%lu", &code, &ttime);
		times[i] = ttime;
		codes[i] = code;

		if (code > _mcode)
			_mcode = code;

		i++;
	}

	*maxcode_out = _mcode;
	return i;
}

HawkesData get_all_data(int tmax){

	unsigned long *times = malloc(MAX_ARRAY_SIZE * sizeof(unsigned long));
	unsigned short int *codes = malloc(MAX_ARRAY_SIZE * sizeof(unsigned short int));
	unsigned int *epar = malloc(MAX_ARRAY_SIZE * sizeof(unsigned int));

	short int maxcode;
	int N;

	N = initialize(times, codes, &maxcode);

	get_earliest_parent(times, N, tmax, epar);

	HawkesData res = {times, codes, epar, N, maxcode};

	return res;
}


int main(){
	HawkesData r = get_all_data(100);

	// printf("%d", r.epar[99]);
}
