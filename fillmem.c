#include <stdio.h>
#include <stdlib.h>

typedef struct  {
	unsigned long *times;
	unsigned short int *codes;
	unsigned int *epar;
	int N;
	short int maxcode;
} HawkesData;

void get_earliest_parent(unsigned long *times, int N, int tmax, unsigned int *epar){

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

int initialize(const char* data_filename, unsigned long *times, unsigned short int *codes, short int *maxcode_out){

	FILE* stream = fopen(data_filename, "r");
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

	fclose(stream);

	*maxcode_out = _mcode;
	return i;
}

int get_line_count(const char * filename){

	int count = 0;
	FILE *handle;
	handle = fopen(filename, "r");

	if (handle == NULL)
		return -1;

	char buf[20];
	while(fgets(buf, sizeof(buf), handle) != NULL)
	  count++;

	return count;
}

HawkesData get_all_data(const char * data_filename, int tmax){

	int array_size = get_line_count(data_filename);

	unsigned long *times = malloc(array_size * sizeof(unsigned long));
	unsigned short int *codes = malloc(array_size * sizeof(unsigned short int));
	unsigned int *epar = malloc(array_size * sizeof(unsigned int));

	short int maxcode;
	int N;

	N = initialize(data_filename, times, codes, &maxcode);

	get_earliest_parent(times, N, tmax, epar);

	HawkesData res = {times, codes, epar, N, maxcode};

	return res;
}

void release_all_data(HawkesData r){
	free(r.times);
	free(r.codes);
	free(r.epar);
}
