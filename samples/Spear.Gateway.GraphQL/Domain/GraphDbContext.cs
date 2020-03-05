﻿using Microsoft.EntityFrameworkCore;

namespace Spear.Gateway.GraphQL.Domain
{
    public class GraphDbContext : DbContext
    {
        public GraphDbContext(DbContextOptions options) : base(options)
        {

        }

        protected override void OnModelCreating(ModelBuilder builder)
        {
            // Set up your relations
        }

        public DbSet<Property> Properties { get; set; }
        public DbSet<PropertyType> PropertyTypes { get; set; }
        public DbSet<Location> Locations { get; set; }
    }

    public class Property
    {
        public uint Id { get; set; }
        public string Name { get; set; }
        public PropertyType Type { get; set; }
        public Location Location { get; set; }
    }

    public class PropertyType
    {
        public uint Id { get; set; }
        public string Name { get; set; }
        public decimal Premium { get; set; }
    }

    public class Location
    {
        public uint Id { get; set; }
        public string Name { get; set; }
    }
}
