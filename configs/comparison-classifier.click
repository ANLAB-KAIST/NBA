
// To compare the overhead of classifier element. (vs comparison-randombranch.click)
// ARP requests are sent to output 0, ARP replies are sent to output 1, IP packets to output 2.
classifier :: Classifier(12/0806 20/0001, 12/0806 20/0002, 12/0800);
IPv4Forwarding3 :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();

DropBroadcasts() -> classifier;

classifier[0] -> Discard();
classifier[1] -> Discard(); 
classifier[2] -> IPv4Forwarding3; 
