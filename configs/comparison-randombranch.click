
// To compare the overhead of classifier element (vs comparison-classifier.click)
randombranch :: RandomWeightedBranch(0, 0, 100);
IPv4Forwarding3 :: CheckIPHeader() -> IPlookup() -> DecIPTTL() -> ToOutput();

DropBroadcasts() -> randombranch;

randombranch[0] -> Discard();
randombranch[1] -> Discard(); 
randombranch[2] -> IPv4Forwarding3; 
