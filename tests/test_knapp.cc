#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <scif.h>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/ctrl.pb.h>
#if 0
#require <engines/knapp/ctrl.pb.o>
#endif

using namespace nba::knapp;

TEST(KnappCommunicationTest, Ping) {
    std::string msg = "hello world";

    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    request.mutable_text()->set_msg(msg);

    char buf[1024];
    uint32_t msgsz = request.ByteSize();
    request.SerializeToArray(buf, msgsz);
    rc = scif_send(sock, &msgsz, sizeof(msgsz), SCIF_SEND_BLOCK);
    ASSERT_EQ(sizeof(msgsz), rc);
    rc = scif_send(sock, buf, msgsz, SCIF_SEND_BLOCK);
    ASSERT_EQ(msgsz, rc);

    rc = scif_recv(sock, &msgsz, sizeof(msgsz), SCIF_RECV_BLOCK);
    ASSERT_EQ(sizeof(msgsz), rc);
    rc = scif_recv(sock, buf, msgsz, SCIF_RECV_BLOCK);
    ASSERT_EQ(msgsz, rc);
    response.ParseFromArray(buf, msgsz);

    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_EQ("hello world", response.text().msg());
    scif_close(sock);
}

// vim: ts=8 sts=4 sw=4 et
