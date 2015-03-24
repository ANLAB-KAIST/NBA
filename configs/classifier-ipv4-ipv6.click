
// IPv4 pkts are sent to output 0, IPv6 pkts are sent to output 1.
classifier :: Classifier(12/0800, 12/86DD);

IPv4Forwarding :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();
IPv6Forwarding :: CheckIP6Header() -> LookupIP6Route() -> DecIP6HLIM() -> ToOutput();

DropBroadcasts() -> classifier;
classifier[0] -> IPv4Forwarding;
classifier[1] -> IPv6Forwarding; 
