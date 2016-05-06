#include <cstdint>
#include <cstdio>
#include <cassert>
#include <gtest/gtest.h>
#include <scif.h>
#include <nba/engines/knapp/defs.hh>
#include <nba/engines/knapp/sharedtypes.hh>
#include <nba/engines/knapp/hosttypes.hh>
#include <nba/engines/knapp/hostutils.hh>
#include <nba/engines/knapp/pollring.hh>
#include <nba/engines/knapp/rma.hh>
#if 0
#require <engines/knapp/ctrl.pb.o>
#require <engines/knapp/hostutils.o>
#require <engines/knapp/pollring.o>
#require <engines/knapp/rma.o>
#endif

using namespace nba::knapp;

class RTEEnvironment : public ::testing::Environment {
    int argc;
    char **argv;

public:
    RTEEnvironment(int argc, char **argv)
        : ::testing::Environment(), argc(argc), argv(argv)
    { }

    void SetUp()
    {
        int rc;
        rc = rte_eal_init(argc, argv);
        assert(rc == 0);
    }

    void TearDown()
    {
        return;
    }
};

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    RTEEnvironment *env = new RTEEnvironment(argc, argv);
    ::testing::AddGlobalTestEnvironment(env);
    return RUN_ALL_TESTS();
}

TEST(KnappCommunicationTest, RawPing) {
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

TEST(KnappCommunicationTest, Ping) {
    std::string msg = "goodbye world";
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    request.mutable_text()->set_msg(msg);

    ctrl_invoke(sock, request, response);

    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_EQ("goodbye world", response.text().msg());

    scif_close(sock);
}

TEST(KnappCommunicationTest, InvalidPing) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::PING);
    // missing text.

    ctrl_invoke(sock, request, response);

    EXPECT_EQ(CtrlResponse::INVALID, response.reply());

    scif_close(sock);
}

TEST(KnappMallocTest, Small) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    request.set_type(CtrlRequest::MALLOC);
    CtrlRequest::MallocParam *m = request.mutable_malloc();
    m->set_size(1024);
    m->set_align(64);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_TRUE(response.has_resource());
    EXPECT_NE(0u, response.resource().handle());
    void *ptr = (void *) (uintptr_t) response.resource().handle();

    request.Clear();
    response.Clear();
    request.set_type(CtrlRequest::FREE);
    request.mutable_resource()->set_handle((uintptr_t) ptr);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());

    scif_close(sock);
}

TEST(KnappMallocTest, Large) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;
    void *ptrs[256];

    for (int i = 0; i < 256; i++) {
        request.Clear();
        response.Clear();
        request.set_type(CtrlRequest::MALLOC);
        CtrlRequest::MallocParam *m = request.mutable_malloc();
        m->set_size(16 * 1024 * 1024);
        m->set_align(64);
        ctrl_invoke(sock, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
        EXPECT_TRUE(response.has_resource());
        EXPECT_NE(0u, response.resource().handle());
        ptrs[i] = (void *) response.resource().handle();
    }

    for (int i = 0; i < 256; i++) {
        request.Clear();
        response.Clear();
        request.set_type(CtrlRequest::FREE);
        request.mutable_resource()->set_handle((uintptr_t) ptrs[i]);
        ctrl_invoke(sock, request, response);
        EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    }

    scif_close(sock);
}

TEST(KnappvDeviceTest, Single) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);

    CtrlRequest request;
    CtrlResponse response;

    request.set_type(CtrlRequest::CREATE_VDEV);
    CtrlRequest::vDeviceInfoParam *v = request.mutable_vdevinfo();
    v->set_num_pcores(1);
    v->set_num_lcores_per_pcore(4);
    v->set_pipeline_depth(32);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());
    EXPECT_TRUE(response.has_resource());
    EXPECT_LE(0u, response.resource().handle());
    void *vdev_handle = (void *) response.resource().handle();

    request.Clear();
    request.set_type(CtrlRequest::DESTROY_VDEV);
    request.mutable_resource()->set_handle((uintptr_t) vdev_handle);
    ctrl_invoke(sock, request, response);
    EXPECT_EQ(CtrlResponse::SUCCESS, response.reply());

    scif_close(sock);
}

TEST(KnappRMATest, H2DWrite) {
    int rc;
    scif_epd_t sock = scif_open();
    struct scif_portID remote = { 1, KNAPP_MASTER_PORT };
    rc = scif_connect(sock, &remote);
    ASSERT_LT(0, rc);
    {
        PollRing r(sock, 15, 0);
        RMABuffer buf(sock, 4096, 0);

        void *ring_va = (void *) r.get_va();
        void *ring_ra = (void *) r.get_ra();
        printf("ring: va=%p, ra=%p\n", ring_va, ring_ra);
        void *buf_va = (void *) buf.get_va();
        void *buf_ra = (void *) buf.get_ra();
        printf("buf: va=%p, ra=%p\n", buf_va, buf_ra);

        EXPECT_FALSE(r.poll(0, 99));
        r.notify(0, 99);
        EXPECT_TRUE(r.poll(0, 99));
        memset(buf_va, 1, sizeof(int));
        buf.write(0, sizeof(int));
        r.remote_notify(0, 99);
    }
    scif_close(sock);
}

// vim: ts=8 sts=4 sw=4 et
