#ifndef __INTERVAL_H
#define __INTERVAL_H


// This is a convenience storage class which will probably be changed
// once we figure out the return type, we should only work with the
// indices here

class Interval{
	public:
		double birth;
		double death;
		size_t birth_index;
		size_t death_index;

		Interval(){}

		// initializations
		Interval(size_t bindex) : birth_index(bindex), death_index(-1){}
		Interval(size_t bindex, double btime) : birth_index(bindex), birth(btime){}

		Interval(size_t bindex, size_t dindex, double btime, double dtime) : birth_index(bindex), death_index(dindex), birth(btime), death(dtime){}

		// close an interval - write doewn a death time
		void close(size_t dindex){
			death_index = dindex;
		}

		// index + death time
		void close(size_t dindex, double dtime){
			death_index = dindex;
			dtime = dtime;
		}
};




#endif
