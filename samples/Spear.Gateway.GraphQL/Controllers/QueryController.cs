using System;
using System.Net;
using EntityGraphQL;
using EntityGraphQL.Schema;
using Microsoft.AspNetCore.Mvc;
using Spear.Gateway.GraphQL.Domain;

namespace Spear.Gateway.GraphQL.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class QueryController : ControllerBase
    {
        private readonly GraphDbContext _dbContext;
        private readonly MappedSchemaProvider<GraphDbContext> _schemaProvider;

        public QueryController(GraphDbContext dbContext, MappedSchemaProvider<GraphDbContext> schemaProvider)
        {
            _dbContext = dbContext;
            _schemaProvider = schemaProvider;
        }

        [HttpPost]
        public object Post([FromBody]QueryRequest query)
        {
            try
            {
                var results = _dbContext.QueryObject(query, _schemaProvider);
                // gql compile errors show up in results.Errors
                return results;
            }
            catch (Exception)
            {
                return HttpStatusCode.InternalServerError;
            }
        }
    }
}