FromInput() ->
LoadBalanceByWeight("from-env") ->
IPsecESPencap() ->
IPsecAES() ->
IPsecAuthHMACSHA1() ->
L2Forward(method echoback) ->
ToOutput();
