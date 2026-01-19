module PS;

export {
    # Create an ID for the new Log stream
    redef enum Log::ID += { LOG };
    
    # Define the datarecord which should be saved to the Log File
    type Features: record {
        ts: time &log;
        uid: string &log;
        id: conn_id &log;
        proto: transport_proto &log;
        up: int &log;
        down: int &log;
        ps: vector of string &log;
    };
}


global pkt_seq: table[string] of vector of int;



# at the startup of zeek create the Log stream
event zeek_init() &priority=5 {
    Log::create_stream(PS::LOG, [$columns=Features, $path="ps"]);
}

# update the measures for each new packet
event new_packet (c: connection, p: pkt_hdr) {

    # set bool if this packet is moving in the fwd direction
    local is_fwd = (p?$ip && p$ip$src == c$id$orig_h || p?$ip6 &&p$ip6$src == c$id$orig_h);
    # bool is true if this packet is tcp
    local is_tcp = p?$tcp;
    # bool is true if this packet is udp
    local is_udp = p?$udp;
    # bool is true if this packet is icmp
    local is_icmp = p?$icmp;
    # bool is true if this packet is icmp
    local is_ip6 = p?$ip6;

    
    if (!(c$uid in pkt_seq)){
       pkt_seq[c$uid] = vector();
    }

    #initialize header size to 0
    local header_size = 0;
    # if it is a tcp packet get the header size from the tcp header
    if( is_tcp ){
        header_size = p$tcp$hl;
    }
    # udp and icmp have a fixed header length of 8
    if( is_udp || is_icmp){
        header_size = 8;
    }
    
    # initialize the payload size to 0
    local data_size = 0;

    # if it is an ip6 packet take the payload size of the ip6 packet and subtract the header size of the encapsulated protocol
    if( is_ip6 ){
        data_size = p$ip6$len - header_size;
    }
    # if it is an ip4 packet take the packet size of the ip4 packet and subtract the ip4 header size and the header size of the encapsulated protocol
    else{
        data_size = p$ip$len - p$ip$hl - header_size;
    }
    
    #pkt_seq[c$uid]
    if(data_size > 0){
    	if ( is_fwd ){
	        pkt_seq[c$uid] += data_size;
	    }
	    # otherwise add it to the bwd vector
	    else {
	        pkt_seq[c$uid] += -data_size;
	    }
    }
}

# if the connection is finished calculate all the features and write them to the log file
event connection_state_remove(c: connection) {
    # fill the Features object for this connection
    local hhh = pkt_seq[c$uid];
    
    local abc: vector of string;
    
    if (|hhh|>0){
    	local up = 0;
    	local down = 0;
    	local tmp_p = hhh[0];
	    local tmp_count = 1;
	    local i = 1;
	    while (i<|hhh|){
	    	if (hhh[i] == tmp_p){
	    		tmp_count += 1;
	    	}
	    	else{
	    		abc += fmt("%d:%d",tmp_p,tmp_count);
	    		if(tmp_p>0){
	    			up += tmp_count;
	    		}
	    		else{
	    			down += tmp_count;
	    		}
	    		tmp_p = hhh[i];
	    		tmp_count = 1;
	    	}
	    	i += 1;
	    }
	    abc += fmt("%d:%d",tmp_p,tmp_count);
	    if(tmp_p>0){
	    	up += tmp_count;
	    }
	    else{
	    	down += tmp_count;
	    }

	    local p = get_port_transport_proto(c$id$resp_p);
	    local rec = PS::Features($ts=c$start_time, $uid = c$uid, $id=c$id, $proto=p, $up = up, $down=down, $ps = abc );
	
	    # delete the still existing table entries of this connection, as they are now not needed any more
	    delete pkt_seq[c$uid];
	    
	    # write the measures of this connection to the log file
	    Log::write(PS::LOG, rec);
    }
    
}
