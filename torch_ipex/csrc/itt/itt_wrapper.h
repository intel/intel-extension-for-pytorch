#ifndef PROFILER_ITT_H
#define PROFILER_ITT_H

void itt_range_push(const char* msg);
void itt_range_pop();
void itt_mark(const char* msg);

#endif // PROFILER_ITT_H
