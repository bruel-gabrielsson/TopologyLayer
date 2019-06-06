#ifndef __INTERVAL_H
#define __INTERVAL_H


// This is a convenience storage class which will probably be changed
// once we figure out the return type, we should only work with the
// indices here

class Interval{
	public:
		float birth;
		float death;
		int birth_index;
		int death_index;

		Interval(){}

		// initializations
		Interval(int bindex) : birth_index(bindex), death_index(-1){}
		Interval(int bindex, float btime) : birth_index(bindex), birth(btime){}

		Interval(int bindex, int dindex, float btime, float dtime) : birth_index(bindex), death_index(dindex), birth(btime), death(dtime){}

		// close an interval - write doewn a death time
		void close(int dindex){
			death_index = dindex;
		}

		// index + death time
		void close(int dindex, float dtime){
			death_index = dindex;
			dtime = dtime;
		}
};




#endif
