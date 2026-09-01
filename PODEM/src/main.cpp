/**********************************************************************/
/*           main function for atpg                */
/*                                                                    */
/*           Author: Bing-Chen (Benson) Wu                            */
/*           last update : 01/21/2018                                 */
/**********************************************************************/

#include "atpg.h"

void usage();

int main(int argc, char *argv[])
{
	string inpFile, vetFile, rlEmbeddingFile, rlActorFile, faultMapFile;
	string rlMode = "backtrace_rl";
	string rlEmbeddingBackend;
	bool rlModeSpecified = false;
	int i, j;
	ATPG atpg; // create an ATPG obj, named atpg

	atpg.timer(stdout, "START");
	atpg.detected_num = 1;
	i = 1;

	/* parse the input switches & arguments */
	while (i < argc)
	{
		// number of test generation attempts for each fault.  used in podem.cpp
		if (strcmp(argv[i], "-anum") == 0)
		{
			atpg.set_total_attempt_num(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-bt") == 0)
		{
			atpg.set_backtrack_limit(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-fsim") == 0)
		{
			vetFile = string(argv[i + 1]);
			atpg.set_fsim_only(true);
			i += 2;
		}
		else if (strcmp(argv[i], "-tdfsim") == 0)
		{
			vetFile = string(argv[i + 1]);
			atpg.set_tdfsim_only(true);
			i += 2;
		}
		else if (strcmp(argv[i], "-tdfatpg") == 0)
		{
			atpg.set_SAF_atpg(false);
			i += 1;
		}
		else if (strcmp(argv[i], "-scoap") == 0)
		{
			atpg.set_SCOAP(true);
			i += 1;
		}
		else if (strcmp(argv[i], "-compression") == 0)
		{
			atpg.set_DTC(true);
			atpg.set_STC(true);
			i += 1;
		}
		else if (strcmp(argv[i], "-dtc") == 0)
		{
			atpg.set_DTC(true);
			i += 1;
		}
		else if (strcmp(argv[i], "-stc") == 0)
		{
			atpg.set_STC(true);
			i += 1;
		}
		else if (strcmp(argv[i], "-flow") == 0)
		{
			atpg.set_flow(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-seed") == 0)
		{
			atpg.set_seed(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-stctime") == 0)
		{
			atpg.set_stc_time(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-stcseed") == 0)
		{
			atpg.set_stc_seed(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-stcmul") == 0)
		{
			atpg.set_stc_mul(atoi(argv[i + 1]));
			i += 2;
		}

		else if (strcmp(argv[i], "-pdxbt") == 0)
		{
			atpg.set_podemx_backtrack_limit(atoi(argv[i + 1]));
			i += 2;
		}
		else if (strcmp(argv[i], "-pdxfail") == 0) // podemx continuous fail limit
		{
			atpg.set_podemx_fail_limit(atoi(argv[i + 1]));
			i += 2;
		}
		// for N-detect fault simulation
		else if (strcmp(argv[i], "-ndet") == 0)
		{
			atpg.detected_num = atoi(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-rl-emb") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "-rl-emb requires a filename\n");
				return EXIT_FAILURE;
			}
			rlEmbeddingFile = string(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-rl-actor") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "-rl-actor requires a filename\n");
				return EXIT_FAILURE;
			}
			rlActorFile = string(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-rl-embedding-backend") == 0)
		{
			if (i + 1 >= argc || (string(argv[i + 1]) != "smartatpg" && string(argv[i + 1]) != "deepgate"))
			{
				fprintf(stderr, "-rl-embedding-backend requires smartatpg or deepgate\n");
				return EXIT_FAILURE;
			}
			rlEmbeddingBackend = string(argv[i + 1]);
			i += 2;
		}
		else if (strcmp(argv[i], "-rl-mode") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "-rl-mode requires a mode\n");
				return EXIT_FAILURE;
			}
			rlMode = string(argv[i + 1]);
			rlModeSpecified = true;
			i += 2;
		}
		else if (strcmp(argv[i], "-fault-map") == 0)
		{
			if (i + 1 >= argc)
			{
				fprintf(stderr, "-fault-map requires a filename\n");
				return EXIT_FAILURE;
			}
			faultMapFile = string(argv[i + 1]);
			i += 2;
		}
		else if (argv[i][0] == '-')
		{
			j = 1;
			while (argv[i][j] != '\0')
			{
				if (argv[i][j] == 'd')
				{
					j++;
				}
				else
				{
					fprintf(stderr, "atpg: unknown option\n");
					usage();
				}
			}
			i++;
		}
		else
		{
			inpFile = string(argv[i]);
			i++;
		}
	}

	/* an input file was not specified, so describe the proper usage */
	if (inpFile.empty())
	{
		usage();
	}
	if (rlEmbeddingFile.empty() != rlActorFile.empty())
	{
		fprintf(stderr, "-rl-emb and -rl-actor must be provided together\n");
		return EXIT_FAILURE;
	}
	if ((rlModeSpecified || !rlEmbeddingBackend.empty()) && rlEmbeddingFile.empty())
	{
		fprintf(stderr, "-rl-mode requires -rl-emb and -rl-actor\n");
		return EXIT_FAILURE;
	}
	try
	{
		atpg.set_rl_mode(rlMode);
	}
	catch (const exception &error)
	{
		fprintf(stderr, "%s\n", error.what());
		return EXIT_FAILURE;
	}

	/* read in and parse the input file */
	atpg.input(inpFile); // input.cpp
	atpg.set_fault_map_path(faultMapFile);
	if (!rlEmbeddingFile.empty())
	{
		try
		{
			atpg.enable_rl_inference(rlEmbeddingFile, rlActorFile, rlEmbeddingBackend);
		}
		catch (const exception &error)
		{
			fprintf(stderr, "Cannot enable RL inference: %s\n", error.what());
			return EXIT_FAILURE;
		}
	}

	/* if vector file is provided, read it */
	if (!vetFile.empty())
	{
		atpg.read_vectors(vetFile);
	}
	atpg.timer(stdout, "for reading in circuit");

	atpg.level_circuit(); // level.cpp
	atpg.timer(stdout, "for levelling circuit");

	atpg.rearrange_gate_inputs(); // level.cpp
	atpg.timer(stdout, "for rearranging gate inputs");

	atpg.create_dummy_gate(); // init_flist.cpp
	atpg.timer(stdout, "for creating dummy nodes");

	try
	{
		if (!atpg.get_tdfsim_only() && atpg.get_SAF_atpg())
			atpg.generate_fault_list(); // init_flist.cpp
		else
			atpg.generate_tdfault_list();
		atpg.timer(stdout, "for generating fault list");
		atpg.test(); // atpg.cpp
		if (!atpg.get_tdfsim_only())
			atpg.compute_fault_coverage(); // init_flist.cpp
	}
	catch (const exception &error)
	{
		fprintf(stderr, "ATPG failed: %s\n", error.what());
		return EXIT_FAILURE;
	}
	atpg.timer(stdout, "for test pattern generation");
	exit(EXIT_SUCCESS);
}

void usage()
{

	fprintf(stderr, "usage: atpg [options] infile\n");
	fprintf(stderr, "Options\n");
	fprintf(stderr, "    -fsim <filename>: fault simulation only; filename provides vectors\n");
	fprintf(stderr, "    -anum <num>: <num> specifies number of vectors per fault\n");
	fprintf(stderr, "    -bt <num>: <num> specifies number of backtracks\n");
	fprintf(stderr, "    -rl-emb <filename>: precomputed DeepGate or SmartATPG descriptors\n");
	fprintf(stderr, "    -rl-embedding-backend <smartatpg|deepgate>: validate artifact backend\n");
	fprintf(stderr, "    -rl-actor <filename>: exported PPO actor weights\n");
	fprintf(stderr, "    -rl-mode <backtrace_rl|propagate_rl|both_rl>: RL decision scope (default: backtrace_rl)\n");
	fprintf(stderr, "    -fault-map <filename>: preserve source collapsed faults on a transformed netlist\n");
	exit(EXIT_FAILURE);

} /* end of usage() */

void ATPG::set_fsim_only(const bool &b)
{
	this->fsim_only = b;
}

void ATPG::set_tdfsim_only(const bool &b)
{
	this->tdfsim_only = b;
}

void ATPG::set_SCOAP(const bool &b)
{
	this->fault_order_by_scoap = b;
}

void ATPG::set_DTC(const bool &b)
{
	this->dynamic_test_compression = b;
}

void ATPG::set_STC(const bool &b)
{
	this->static_test_compression = b;
}

void ATPG::set_flow(const int &i)
{
	this->flow = i;
}
void ATPG::set_stc_time(const int &i)
{
	this->stctime = i;
}
void ATPG::set_stc_seed(const int &i)
{
	this->stcseed = i;
}
void ATPG::set_stc_mul(const int &i)
{
	this->stcmul = i;
}

void ATPG::set_podemx_backtrack_limit(const int &i)
{
	this->podemx_backtrack_limit = i;
}
void ATPG::set_podemx_fail_limit(const int &i)
{
	this->fail_continuous_limit = i;
}
