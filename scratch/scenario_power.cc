#include "ns3/core-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"
#include <cmath>
#include <random>
#include <tuple>
#include "ns3/netanim-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/flow-monitor-helper.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include<string.h>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("NetworkScenario");

class NetworkScenario
{
    public:
        void initialize(
            int num_enb,
            std::vector<std::vector<int>> enb_position,
            std::vector<int> enb_power,
            std::vector<int> ue_per_enb,
            int it_period,
            int sim_time);
        void run();

        void enable_trace();
    protected:
        int num_enb;
        std::vector<std::vector<int>> enb_position;
        std::vector<int> enb_power;
        std::vector<int> ue_per_enb;
        int it_period;
        int sim_time;

        Ptr<FlowMonitor> Monitor;
        FlowMonitorHelper flowmon;
        ns3::Time last_ue_arrival;
        int min_power;
        int max_power;
        void ue_depart_callback();
        void ue_arrive_callback();

        NodeContainer enb_nodes;
        NodeContainer ue_nodes;
        NetDeviceContainer enb_devices;

        Ptr<LteHelper> lte_helper;
        Ptr<EpcHelper> epc_helper;
        NodeContainer server_nodes;
        std::vector<PacketSizeMinMaxAvgTotalCalculator*> ue_packet_calcs;

        void create_enb_nodes();
        void create_ue_nodes();

        void create_lte_network();
        void apply_network_conf();
        void create_remote_server();
        void create_ue_applications();
        static void callback_ipv4_packet_received(PacketSizeMinMaxAvgTotalCalculator* packet_calc,Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface);
        void callback_ue_spotted_at_enb(std::string context, const uint64_t imsi, const uint16_t cell_id, const uint16_t rnti);
        void callback_measurement_report_received(const uint64_t imsi, const uint16_t cell_id,  const uint16_t rnti, const LteRrcSap::MeasurementReport report);
        void setup_callbacks();
        void printStats(FlowMonitorHelper &flowmonHelper, bool perFlowInfo);

        void dump_initial_state();
        void periodically_interact_with_agent();

         int timestep() { return Simulator::Now().GetMilliSeconds();  }

};

void NetworkScenario::initialize(
            int num_enb,
            std::vector<std::vector<int>> enb_position,
            std::vector<int> enb_power,
            std::vector<int> ue_per_enb,
            int it_period,
            int sim_time)

    {
        this->num_enb = num_enb;
        this->enb_position = enb_position;
        this->enb_power = enb_power;
        this->ue_per_enb = ue_per_enb;
        this->it_period = it_period;
        this->sim_time = sim_time;
        this->min_power = 19;
        this->max_power = 65;

        this->create_enb_nodes();
        this->create_ue_nodes();

        this->create_lte_network();
        this->apply_network_conf();
        this->create_remote_server();
        this->create_ue_applications();
       // this->setup_callbacks();
}

void NetworkScenario::run(){
    // this->dump_initial_state();

    mkfifo("fifo1", 0666);
    mkfifo("fifo2", 0666);

    this->periodically_interact_with_agent();

    this->Monitor = this->flowmon.Install(this->ue_nodes);
    this->Monitor = this->flowmon.Install(this->server_nodes);

    // AnimationInterface anim ("wireless-animation.xml"); // Mandatory
    // anim.SetMaxPktsPerTraceFile(0xFFFFFFFF);

    Simulator::Stop(Seconds(sim_time));
    Simulator::Run();

    this->printStats(this->flowmon,true);
    Simulator::Destroy();
}

void NetworkScenario::create_enb_nodes(){
    this->enb_nodes.Create(num_enb);
    MobilityHelper mobility_helper;
    mobility_helper.SetMobilityModel("ns3::ConstantPositionMobilityModel"); // Explicit model
    mobility_helper.Install(this->enb_nodes);

    for (auto i = 0; i < num_enb; i++)
    {
        Ptr<Node> enb_node = this->enb_nodes.Get(i);
        Ptr<MobilityModel> mobility = enb_node->GetObject<MobilityModel>();
        mobility->SetPosition(Vector(this->enb_position[i][0], this->enb_position[i][1], this->enb_position[i][2])); // Fixed Z-value
    }
}


void NetworkScenario::create_ue_nodes()
{
    for (auto i = 0; i < num_enb; i++) {
        NodeContainer ue_nodes_per_enb;
        ue_nodes_per_enb.Create(this->ue_per_enb[i]);
        this->ue_nodes.Add(ue_nodes_per_enb);

        // Create position allocator for UEs around eNB
        Ptr<UniformDiscPositionAllocator> uePositionAlloc = CreateObject<UniformDiscPositionAllocator>();
        uePositionAlloc->SetX(this->enb_position[i][0]);
        uePositionAlloc->SetY(this->enb_position[i][1]);
        uePositionAlloc->SetRho(500.0);  // Max distance from eNB

        // Create a random speed variable
        Ptr<UniformRandomVariable> speed = CreateObject<UniformRandomVariable>();
        speed->SetAttribute("Min", DoubleValue(2.0));  // Min speed: 2 m/s
        speed->SetAttribute("Max", DoubleValue(4.0));  // Max speed: 4 m/s

        // Set mobility model for UEs
        MobilityHelper mobility_helper;
        mobility_helper.SetMobilityModel("ns3::RandomWalk2dOutdoorMobilityModel",
            "Speed", PointerValue(speed),
            "Bounds", RectangleValue(Rectangle(0, 5000, 0, 5000))); // Adjusted to simulation area

        mobility_helper.SetPositionAllocator(uePositionAlloc);
        mobility_helper.Install(ue_nodes_per_enb);
    }
}

void NetworkScenario::setup_callbacks()
{
     // Create packet calculators for each UE, and set up callbacks to count the
     // number of bytes received over IPv4. Used by get_ue_rx_bytes() below
     for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
         PacketSizeMinMaxAvgTotalCalculator *packet_calc = new PacketSizeMinMaxAvgTotalCalculator();
         this->ue_nodes.Get(i)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext("Rx", MakeBoundCallback(&NetworkScenario::callback_ipv4_packet_received, packet_calc));
         this->ue_packet_calcs.push_back(packet_calc);
     }

     // Connect callbacks to trigger whenever a UE is connected to a new eNodeB,
     // either because of initial network attachment or because of handovers
     Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
         MakeCallback(&NetworkScenario::callback_ue_spotted_at_enb, this));
     Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
         MakeCallback(&NetworkScenario::callback_ue_spotted_at_enb, this));

     // Connect callback for whenever an eNodeB receives "measurement reports".
     // These reports contain signal strength information of neighboring cells,
     // as seen by a UE. This is used by the eNodeB to determine handovers
     Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/RecvMeasurementReport",
         MakeCallback(&NetworkScenario::callback_measurement_report_received, this));
}

void NetworkScenario::callback_ue_spotted_at_enb(
        std::string context, const uint64_t imsi,
        const uint16_t cell_id, const uint16_t rnti)
{
    // A given eNodeB (identified by cell ID) has become responsible for an UE
    // (identified by its IMSI), due to initial network attachment or handover
    std::cout << this->timestep() << " ms: UE seen at cell: "<< "Cell " << (int)cell_id << " saw IMSI " << imsi << std::endl;
}
void NetworkScenario::callback_ipv4_packet_received(
        PacketSizeMinMaxAvgTotalCalculator* packet_calc,
        Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface)
{
    // Callback for each packet received at the IPv4 layer. Pass the packet
    // directly on to the PacketSizeMinMaxAvgTotalCalculator, which is used by
    // the periodic UE state reporting method via this->get_ue_rx_bytes()
    packet_calc->PacketUpdate("", packet);
}
void NetworkScenario::callback_measurement_report_received(
        const uint64_t imsi, const uint16_t cell_id,
        const uint16_t rnti, const LteRrcSap::MeasurementReport report)
{
    // An eNodeB has received a measurement report of neighboring cell signal
    // strengths from an attached UE. Dump interesting information to stdout
    // std::cout << this->timestep() << " ms: Measurement report: "
    //     << "Cell " << (int)cell_id
    //     << " got report from IMSI " << imsi
    //     << ": " << (int)cell_id
    //     << "/" << (int)report.measResults.rsrpResult
    //     << "/" << (int)report.measResults.rsrqResult;

    // There might be additional measurements to the one listed directly in the
    // data structure, hence we need to do some additional iteration
    for (auto iter = report.measResults.measResultListEutra.begin();
            iter != report.measResults.measResultListEutra.end(); iter++) {
        std::cout << " " << (int)iter->physCellId << "/"
            << (int)iter->rsrpResult << "/" << (int)iter->rsrqResult;
    }
    std::cout << std::endl;
}


void NetworkScenario::ue_depart_callback(){
    ns3::Time now = Simulator::Now();

    std::cout<<this->timestep()<<" ms UE has departed \n";
    Simulator::Schedule(now+ns3::Time("1s"), &NetworkScenario::ue_arrive_callback, this);
}

void NetworkScenario::ue_arrive_callback(){
    ns3::Time now = Simulator::Now();
    std::cout<<this->timestep()<<" ms UE has departed \n";
    Simulator::Schedule( now+ns3::Time("1s"),&NetworkScenario::ue_depart_callback, this);
}

void NetworkScenario::dump_initial_state()
{
    // Upon start of simulation, dump position and orientation of each eNodeB
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        Ptr<Node> node = this->enb_nodes.Get(i);
        Vector position = node->GetObject<MobilityModel>()->GetPosition();
        std::cout << this->timestep() << " ms: Cell state: "
            << "Cell " << (i + 1)
            << " at " << position.x << " " << position.y
            << std::endl;
    }

    // Also dump the random number generator seed used to generate UE clusters
    std::cout << this->timestep() << " ms: Seed " << std::endl;
}

void NetworkScenario::periodically_interact_with_agent()
{
    // Dump relevant simulation state for each UE to stdout. Currently we are
    // interested in 2D position and IPv4 bytes received since last time
    // for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
    //     Ptr<Node> node = this->ue_nodes.Get(i);
    //     Vector position = node->GetObject<MobilityModel>()->GetPosition();
    //     std::cout << this->timestep() << " ms: UE state: "    << "IMSI " << (i + 1)    << " at " << position.x << " " << position.y<< std::endl;
    // }

    int fd1 = open("fifo1", O_WRONLY);
    int fd2 = open("fifo2", O_RDONLY);

    if (fd1 == -1 || fd2 == -1) {
        std::cerr << "Error: Could not open FIFOs" << std::endl;
        return;
    }

    std::cout<<"Starting to send \n";
    std::stringstream ss;
    // fflush(stdout);

    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        ss << this->enb_power[i] << ",";
    }
    std::string tx_power_str = ss.str();
    std::cout << "Current Tx Power: " << tx_power_str << std::endl;

    // Send the power levels through FIFO1
    write(fd1, tx_power_str.c_str(), tx_power_str.size());

    char rbuf[50];
    memset(rbuf, 0, sizeof(rbuf));

    // Read from FIFO2 (DRL agent response)
    ssize_t bytesRead = read(fd2, rbuf, sizeof(rbuf) - 1);
    if (bytesRead > 0) {
        rbuf[bytesRead] = '\0';  // Null-terminate to avoid garbage characters
        std::cout << "Received new Tx Power: " << rbuf << std::endl;
    } else {
        std::cerr << "Error: No data received from DRL agent" << std::endl;
        close(fd1);
        close(fd2);
        return; // No valid data, exit early
    }

    // Parse received power values
    std::vector<int> recv_power;
    std::stringstream power_stream(rbuf);
    std::string token;

    while (std::getline(power_stream, token, ',')) {
        try {
            recv_power.push_back(std::stoi(token));
        } catch (std::exception &e) {
            std::cerr << "Error parsing power value: " << token << std::endl;
        }
    }

    // Update eNB power values
    if (this->timestep() >= 200 && recv_power.size() >= this->enb_nodes.GetN() - 1) {
        int power;
        for (uint32_t i = 1; i < this->enb_nodes.GetN(); i++) {  // Start from index 1
            power = recv_power[i - 1];  // Get the received power for the current eNB (shifted index)

            // Check if the received power is outside the acceptable range
            if (power < this->min_power) {
                power = this->min_power;  // Set to minimum if below range
            } else if (power > this->max_power) {
                power = this->max_power;  // Set to maximum if above range
            }

            this->enb_power[i] = power;  // Set the power for the eNB
        }

        // Apply the network configuration after power adjustment
        this->apply_network_conf();
    }

    close(fd1);
    close(fd2);

    // Reschedule again after this->interaction_interval (default 100 ms)
    Simulator::Schedule(MilliSeconds(it_period),&NetworkScenario::periodically_interact_with_agent, this);
}


void NetworkScenario::enable_trace()
{
    this->lte_helper->EnableTraces();
}

void NetworkScenario::create_lte_network()
{
    // Create LTE and EPC helpers. Network to be set up as a bunch of LTE base
    // stations (eNodeB), attached to an EPC (network core) implementation and
    // UEs (mobile handsets)
    this->epc_helper = CreateObject<PointToPointEpcHelper>();
    this->lte_helper = CreateObject<LteHelper>();
    this->lte_helper->SetEpcHelper(this->epc_helper);
    this->lte_helper->SetAttribute ("NumberOfComponentCarriers", UintegerValue (1));
    this->lte_helper->SetAttribute ("EnbComponentCarrierManager", StringValue ("ns3::RrComponentCarrierManager"));
    this->lte_helper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (100));
    this->lte_helper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (100));

    // Set up a directional antenna, to allow 3-sector base stations
    // this->lte_helper->SetEnbAntennaModelType("ns3::ParabolicAntennaModel");
    // this->lte_helper->SetEnbAntennaModelAttribute("Beamwidth", DoubleValue(70.0));

    // Activate handovers using a default RSRQ-based algorithm
    this->lte_helper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");
    this->lte_helper->SetHandoverAlgorithmAttribute("ServingCellThreshold", UintegerValue(30));


    // Select "hard" frequency reuse (FR), which fully partitions the spectrum
    // into three equal parts and distributes those among the base stations
    this->lte_helper->SetFfrAlgorithmType("ns3::LteFrHardAlgorithm");
    // this->lte_helper->SetEnbComponentCarrierManagerType("ns3::RrComponentCarrierManager");


    // Config::SetDefault ("ns3::LteHelper::UseCa", BooleanValue (true));
   // Config::SetDefault ("ns3::LteHelper::EnbComponentCarrierManager", StringValue ("ns3::RrComponentCarrierManager"));

    // Specify that the RLC layer of the LTE stack should use Acknowledged Mode
    // (AM) as the default mode for all data bearers. This as opposed to the
    // ns-3 default which is Unacknowledged Mode (UM), see lte-enb-rrc.cc:1699
    // and lte-helper.cc:618. This is important because TCP traffic between a
    // UE and a remote host is very sensitive to packet loss. Packets lost
    // between the UE and eNodeB will be treated by TCP as a signal that the
    // network is congested -- but it might simply be that the radio conditions
    // are bad! RLC AM mode ensures reliable delivery across the radio link,
    // relieving TCP of that responsibility and not triggering any congestion
    // control algorithms in TCP. This greatly improves TCP performance
    // Config::SetDefault("ns3::LteEnbRrc::EpsBearerToRlcMapping",
    //     EnumValue(LteEnbRrc::RLC_AM_ALWAYS));

    // Bump the maximum possible number of UEs connected per eNodeB
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(80));

    // Loop through the eNodeB nodes and set up the base stations
    // for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
    //     Ptr<Node> node = this->enb_nodes.Get(i);
    //     this->lte_helper->SetFfrAlgorithmAttribute(
    //         "FrCellTypeId", UintegerValue((i % 3) + 1));
    //     this->enb_devices.Add(this->lte_helper->InstallEnbDevice(node));
    // }
    this->enb_devices=this->lte_helper->InstallEnbDevice(this->enb_nodes);

    // Add an X2 interface between the eNodeBs, to enable handovers
    this->lte_helper->AddX2Interface(this->enb_nodes);

}

void NetworkScenario::apply_network_conf()
{
    // Set base station transmission powers according to chosen values
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        std::ostringstream oss;
        oss << "/NodeList/" << this->enb_nodes.Get(i)->GetId();
        oss << "/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/TxPower";
        Config::Set(oss.str(), DoubleValue(0.1 * this->enb_power[i]));
    }
}

void NetworkScenario::create_remote_server()
{
    // Create the server that will send downlink traffic to UEs
    this->server_nodes.Create(1);
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->server_nodes);

    // Connect the server to the PDN gateway (PGW) in the EPC
    PointToPointHelper p2p_helper;
    p2p_helper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2p_helper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer server_devices = p2p_helper.Install(
        this->server_nodes.Get(0), this->epc_helper->GetPgwNode());

    // Set up IP interfaces on the link between PGW and the server
    Ipv4AddressHelper ipv4_helper("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer server_ifaces = ipv4_helper.Assign(server_devices);

    // Add an IP route on the server toward the PGW interface, for the UE subnet (7.0.0.0/8)
    Ipv4StaticRoutingHelper routing_helper;
    Ptr<Ipv4StaticRouting> server_routing = routing_helper.GetStaticRouting(
        this->server_nodes.Get(0)->GetObject<Ipv4>());
    int server_iface_toward_pgw = server_ifaces.Get(0).second;
    server_routing->AddNetworkRouteTo("7.0.0.0", "255.0.0.0", server_iface_toward_pgw);
}

void NetworkScenario::create_ue_applications()
{
    // Add a default IP stack to the UEs. The EPC helper will later
    // assign addresses to UEs in the 7.0.0.0/8 subnet by default
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->ue_nodes);

    // Create UE net devices and assign IP addresses in EPC
    NetDeviceContainer ue_devices = this->lte_helper->InstallUeDevice(this->ue_nodes);
    Ipv4InterfaceContainer ue_ifaces = this->epc_helper->AssignUeIpv4Address(ue_devices);


    // auto ueDev = DynamicCast<LteUeNetDevice> (ue_devices.Get(0));

    // std::map< uint8_t, Ptr<ComponentCarrierUe> > ueCcMap = ueDev->GetCcMap ();
    // ueDev->SetDlEarfcn (ueCcMap.at (1)->GetDlEarfcn());
    // this->lte_helper->Attach(ueDev);
    // Attach the UEs to the LTE network
    this->lte_helper->Attach(ue_devices);

    // Set default IP route for all UEs
    Ipv4StaticRoutingHelper routing_helper;
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        routing_helper.GetStaticRouting(this->ue_nodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(this->epc_helper->GetUeDefaultGatewayAddress(), 1);
    }

    // Set up CBR (constant bitrate) traffic generators from the server to each UE
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        const char *socket_factory_type = "ns3::TcpSocketFactory";
        InetSocketAddress cbr_dest(ue_ifaces.GetAddress(i), 10000);
        OnOffHelper cbr_helper(socket_factory_type, cbr_dest);
        cbr_helper.SetConstantRate(DataRate("20Mbps"));
        ApplicationContainer cbr_apps = cbr_helper.Install(this->server_nodes.Get(0));
        cbr_apps.Start(Seconds(1));

        // Set up a TCP/UDP sink on the receiving side (UE)
        PacketSinkHelper packet_sink_helper(socket_factory_type, cbr_dest);
        ApplicationContainer sink_apps = packet_sink_helper.Install(this->ue_nodes.Get(i));
        sink_apps.Start(Seconds(0));
    }
}

 void NetworkScenario::printStats(FlowMonitorHelper &flowmon_helper, bool perFlowInfo)
{
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmon_helper.GetClassifier());
    std::string proto;
    Ptr<FlowMonitor> monitor = flowmon_helper.GetMonitor ();
    std::map < FlowId, FlowMonitor::FlowStats > stats = monitor->GetFlowStats();
    double totalTimeReceiving;
    uint64_t totalPacketsReceived, totalPacketsDropped, totalBytesReceived;

    totalBytesReceived = 0, totalPacketsDropped = 0, totalPacketsReceived = 0, totalTimeReceiving = 0;
    for (std::map< FlowId, FlowMonitor::FlowStats>::iterator flow = stats.begin(); flow != stats.end(); flow++)
    {
      Ipv4FlowClassifier::FiveTuple  t = classifier->FindFlow(flow->first);
      switch(t.protocol)
       {
       case(6):
           proto = "TCP";
           break;
       case(17):
           proto = "UDP";
           break;
       default:
           exit(1);
       }
       totalBytesReceived += (double) flow->second.rxBytes * 8;
       totalTimeReceiving += flow->second.timeLastRxPacket.GetSeconds ();
       totalPacketsReceived += flow->second.rxPackets;
       totalPacketsDropped += flow->second.txPackets - flow->second.rxPackets;
       if (perFlowInfo) {
         std::cout << "FlowID: " << flow->first << " (" << proto << " "
                   << t.sourceAddress << " / " << t.sourcePort << " --> "
                   << t.destinationAddress << " / " << t.destinationPort << ")" << std::endl;
         std::cout << "  Tx Bytes: " << flow->second.txBytes << std::endl;
         std::cout << "  Rx Bytes: " << flow->second.rxBytes << std::endl;
         std::cout << "  Tx Packets: " << flow->second.txPackets << std::endl;
         std::cout << "  Rx Packets: " << flow->second.rxPackets << std::endl;
         std::cout << "  Time LastRxPacket: " << flow->second.timeLastRxPacket.GetSeconds () << "s" << std::endl;
         std::cout << "  Lost Packets: " << flow->second.lostPackets << std::endl;
         std::cout << "  Pkt Lost Ratio: " << ((double)flow->second.txPackets-(double)flow->second.rxPackets)/(double)flow->second.txPackets << std::endl;
         std::cout << "  Throughput: " << ( ((double)flow->second.rxBytes*8) / (flow->second.timeLastRxPacket.GetSeconds ()) ) << "bps" << std::endl;
         std::cout << "  Mean{Delay}: " << (flow->second.delaySum.GetSeconds()/flow->second.rxPackets) << std::endl;
         std::cout << "  Mean{Jitter}: " << (flow->second.jitterSum.GetSeconds()/(flow->second.rxPackets)) << std::endl;
       }
     }
}


static ns3::GlobalValue g_num_enb ("num_enb", "Number of eNBs",
                                   ns3::UintegerValue (4),
                                   ns3::MakeUintegerChecker<uint32_t> ());

static ns3::GlobalValue g_ue_per_enb ("ue_per_enb", "Number of UEs per eNB",
                                      ns3::UintegerValue (3),
                                      ns3::MakeUintegerChecker<uint32_t> ());

static ns3::GlobalValue g_it_interval ("it_period", "Period to run callbacks to interact with DRL agent in ms",
                                       ns3::UintegerValue (100),
                                       ns3::MakeUintegerChecker<uint32_t> ());

static ns3::GlobalValue g_sim_time ("sim_time", "Simulation Time in s",
                                     ns3::UintegerValue (30),
                                     ns3::MakeUintegerChecker<uint32_t> ());

int main(int argc, char *argv[])
{
    // Command Line arguments
    CommandLine cmd;
    cmd.Parse(argc, argv);

    // The maximum X and Y coordinates of the scenario
    double maxXAxis = 5000;
    double maxYAxis = 5000;

    // Retrieve Global values
    UintegerValue uintegerValue;

    GlobalValue::GetValueByName("num_enb", uintegerValue);
    uint32_t num_enb = uintegerValue.Get();

    GlobalValue::GetValueByName("ue_per_enb", uintegerValue);
    uint32_t ue_per_enb = uintegerValue.Get();

    GlobalValue::GetValueByName("it_period", uintegerValue);
    uint32_t it_interval = uintegerValue.Get();

    GlobalValue::GetValueByName("sim_time", uintegerValue);
    uint32_t sim_time = uintegerValue.Get();

    // Define the center position
    Vector centerPosition(maxXAxis / 2, maxYAxis / 2, 3);

    // Define a list to hold eNB positions
    std::vector<std::vector<int>> enb_position;

    if (num_enb % 2 == 1) // Odd case: One eNB at the center
    {
        enb_position.push_back({static_cast<int>(centerPosition.x), static_cast<int>(centerPosition.y), 3});
    }

    // Distribute remaining eNBs in a circle
    double radius = 1000; // Distance from the center
    for (uint32_t i = 0; i < num_enb-1; i++)
    {
        double angle = (2 * M_PI * i) / num_enb; // Equally spaced angles
        int x = static_cast<int>(centerPosition.x + radius * cos(angle));
        int y = static_cast<int>(centerPosition.y + radius * sin(angle));

        // Ensure positions stay within bounds
        x = std::max(0, std::min(static_cast<int>(maxXAxis), x));
        y = std::max(0, std::min(static_cast<int>(maxYAxis), y));

        enb_position.push_back({x, y, 3});
    }

    std::vector<int> enb_power(num_enb, 60);
    std::vector<int> vector_ue_per_enb(num_enb, ue_per_enb);

    // Initialize network scenario
    NetworkScenario *scenario;
    scenario = new NetworkScenario();

    scenario->initialize(num_enb, enb_position, enb_power, vector_ue_per_enb, it_interval, sim_time);
    scenario->run();

    return 0;

}
