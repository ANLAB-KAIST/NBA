

// ARP requests are sent to output 0, ARP replies are sent to output 1, IP packets to output 2.
classifier :: Classifier(12/0806 20/0001, 12/0806 20/0002, 12/0800);
IPv4Forwarding1 :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();
IPv4Forwarding2 :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();
IPv4Forwarding3 :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();

DropBroadcasts() -> classifier;

//classifier[0] -> Discard();
//classifier[1] -> Discard(); 
classifier[0] -> IPv4Forwarding1;
classifier[1] -> IPv4Forwarding2; 
classifier[2] -> IPv4Forwarding3; 
