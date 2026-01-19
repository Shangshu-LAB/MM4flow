module RAW;

export {
	# Create an ID for the new Log stream
	redef enum Log::ID += { LOG };

	# Define the datarecord which should be saved to the Log File
	type Features: record {
		ts: time &log;
		uid: string &log;
		id: conn_id &log;
		proto: transport_proto &log;
		fwd_raw: string &log;
		bwd_raw: string &log;
	};
}

redef udp_content_deliver_all_orig = T;
redef udp_content_deliver_all_resp = T;
redef tcp_content_deliver_all_orig = T;
redef tcp_content_deliver_all_resp = T;

global flow_fwd_raw: table[string] of string;
global flow_bwd_raw: table[string] of string; 

global raw_maxlen = 256;


function string_to_hex(s: string): string {  
    local result: string = "";  
    for (item in s) {  
        result += fmt("%02x", bytestring_to_count(item));  
    }  
    return result; 
}  

# at the startup of zeek create the Log stream
event zeek_init() &priority=5
	{
	Log::create_stream(RAW::LOG, [ $columns=Features, $path="raw" ]);
	}

event tcp_contents(c: connection, is_orig: bool, seq: count, contents: string){
	if (!(c$uid in flow_fwd_raw)){
		flow_fwd_raw[c$uid] = "";
		flow_bwd_raw[c$uid] = "";
	}
	if (is_orig){
		if (|flow_fwd_raw[c$uid]| < raw_maxlen){
			flow_fwd_raw[c$uid] += contents[0 : raw_maxlen - |flow_fwd_raw[c$uid]|];
		}
	}
	else{
		if (|flow_bwd_raw[c$uid]| < raw_maxlen){
			flow_bwd_raw[c$uid] += contents[0 : raw_maxlen - |flow_bwd_raw[c$uid]|];
		}
	}
}

event udp_contents(c: connection, is_orig: bool, contents: string){
	if (!(c$uid in flow_fwd_raw)){
		flow_fwd_raw[c$uid] = "";
		flow_bwd_raw[c$uid] = "";
	}
	if (is_orig){
		if (|flow_fwd_raw[c$uid]| < raw_maxlen){
			flow_fwd_raw[c$uid] += contents[0 : raw_maxlen - |flow_fwd_raw[c$uid]|];
		}
	}
	else{
		if (|flow_bwd_raw[c$uid]| < raw_maxlen){
			flow_bwd_raw[c$uid] += contents[0 : raw_maxlen - |flow_bwd_raw[c$uid]|];
		}
	}
}

# if the connection is finished calculate all the features and write them to the log file
event connection_state_remove(c: connection)
	{
	# fill the Features object for this connection

	if (c$uid in flow_fwd_raw)
		{
		local p = get_port_transport_proto(c$id$resp_p);
		local rec = RAW::Features($ts=c$start_time, $uid=c$uid, $id=c$id, $proto=p, $fwd_raw=string_to_hex(flow_fwd_raw[c$uid]), $bwd_raw=string_to_hex(flow_bwd_raw[c$uid]));

		# delete the still existing table entries of this connection, as they are now not needed any more
		delete flow_fwd_raw[c$uid];
		delete flow_bwd_raw[c$uid];

		# write the measures of this connection to the log file
		Log::write(RAW::LOG, rec);
		}
	}
