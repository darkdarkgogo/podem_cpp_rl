#include "atpg.h"

void ATPG::set_SAF_atpg(const bool &enabled)
{
	this->SAF_atpg = enabled;
}

void ATPG::set_total_attempt_num(const int &attempts)
{
	this->total_attempt_num = attempts;
}

void ATPG::set_backtrack_limit(const int &limit)
{
	this->backtrack_limit = limit;
}

void ATPG::set_seed(const int &value)
{
	this->seed = value;
}
