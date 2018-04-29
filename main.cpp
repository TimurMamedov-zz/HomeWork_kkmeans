#include <iostream>
#include <unordered_map>

#include <boost/tokenizer.hpp>

#include <dlib/clustering.h>
#include <dlib/image_io.h>
#include <dlib/image_transforms.h>

using sample_type = dlib::matrix<double, 2, 1>;
using kernel_type = dlib::linear_kernel<sample_type>;
using clusters_type = std::unordered_map<std::size_t, std::vector<sample_type> >;

void saveImage(const clusters_type& clusters,
               std::size_t img_size, std::size_t scale, std::size_t N)
{
    dlib::array2d<dlib::hsi_pixel> img(img_size, img_size);

    double color = 255.0 / N;

    for(auto& cluster : clusters)
    {
        auto h = cluster.first * color;

        for(const auto& sample : cluster.second)
        {
            auto x = static_cast<std::size_t>((sample(0) + 100) * scale) % img_size;
            auto y = static_cast<std::size_t>((sample(1) + 100) * scale) % img_size;

            img[y][x] = dlib::hsi_pixel(h, 0xff, 0x7f);
        }
    }

    dlib::save_png(img, "./kmeans.png");
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout << "Usage: kkmeans N img_size scale" << std::endl;
        return 0;
    }

    std::size_t N = std::stol(argv[1]);
    std::size_t img_size = argc > 2 ? std::atol(argv[2]) : 200;
    std::size_t scale = argc > 3 ? std::atol(argv[3]) : 1;

    dlib::kcentroid<kernel_type> kc(kernel_type(), 0.01, 8);
    dlib::kkmeans<kernel_type> test(kc);

    std::vector<sample_type> samples;
    std::vector<sample_type> initial_centers;

    std::string line;

    while(std::getline(std::cin, line))
    {
        std::vector<std::string> tokens;

        boost::char_separator<char> sep{";\n", " "};
        boost::tokenizer<boost::char_separator<char>> tok{line, sep};
        std::copy( tok.begin(), tok.end(), std::back_inserter(tokens) );

        sample_type m;
        m(0) = std::stof(tokens[0]);
        m(1) = std::stof(tokens[1]);

        samples.emplace_back(std::move(m));
    }

    test.set_number_of_centers(N);
    dlib::pick_initial_centers(test.number_of_centers(), initial_centers, samples, test.get_kernel());
    dlib::find_clusters_using_kmeans(samples, initial_centers);
    test.train(samples, initial_centers);

    clusters_type clusters;

    std::ofstream kmeans_train("kmeans.txt");
    for(const auto& sample: samples)
    {
        auto cluster = test(sample);
        kmeans_train << sample(0) << ";" << sample(1) << ";" << cluster << "\n";
        clusters[cluster].push_back(sample);
    }
    kmeans_train.close();

    saveImage(clusters, img_size, scale, N);

    return 0;
}
